import gc
import json
import os
import sys
import platform
import gzip
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, TypedDict
from pympler import muppy, summary, asizeof

import numpy as np
import psutil
from constants import WORKERS, NOT_ARMY, VALID_UNITS
from sc2.constants import CREATION_ABILITY_FIX
from sc2.data import Race, race_worker
from sc2.game_state import GameState
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.main import run_replay
from sc2.observer_ai import ObserverAI
from sc2.unit import Unit

from models import UnitLifetime, UnitPosition

MAX_GRID_SIZE = 182
NUMBER_OF_GRIDS = 20
SIZE_OF_GRID = MAX_GRID_SIZE / NUMBER_OF_GRIDS
NUM_PREDICTED_STEPS = 2


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1024 ** 2:.2f} MB")  # Resident Set Size


def get_unit_type(unit_name: str):
    unit_name = unit_name.strip().upper()
    for race, unit_set in VALID_UNITS.items():
        for unit_type in unit_set:
            if unit_type.name == unit_name:
                return unit_type
    return None


class StepData(TypedDict):
    iteration: int
    visibility: List
    enemy_units_seen_and_alive: Dict[int, Dict]
    player_pov: int
    player_units: Dict[int, Dict]
    workers_built: int
    army_built: int
    supply_cap: int
    supply_left: int
    supply_used: int
    supply_army: int
    supply_workers: int
    own_race: Race
    enemy_race: Race
    own_spawn_x: float
    own_spawn_y: float
    enemy_spawn_x: float
    enemy_spawn_y: float
    minerals: int
    gas: int
    under_construction: Dict[int, int]
    newly_queued: Dict[int, int]
    building_started: Dict[int, int]
    lair_started: int
    hive_started: int


# todo include compressed data on units/buildings constructed in past step block into a single output
def compress_step_block(steps: List[StepData]) -> StepData:
    base = steps[-1].copy()  # Use the latest frame as a base

    # Aggregate scalar values
    base["workers_built"] = sum(s["workers_built"] for s in steps)
    base["army_built"] = sum(s["army_built"] for s in steps)
    accum = defaultdict(int)
    for s in steps:
        for k, v in s["newly_queued"].items():
            accum[k] += v
    base["newly_queued"] = dict(accum)

    accum = defaultdict(int)
    for s in steps:
        for k, v in s["under_construction"].items():
            accum[k] += v
    base["under_construction"] = dict(accum)

    # Optionally average or use max of supply/minerals if desired
    base["minerals"] = max(s["minerals"] for s in steps)
    base["gas"] = max(s["gas"] for s in steps)

    # Merge enemy_units_seen_and_alive by updating to latest sighting
    enemy_units = {}
    for s in steps:
        enemy_units.update(s["enemy_units_seen_and_alive"])
    base["enemy_units_seen_and_alive"] = enemy_units

    accum = defaultdict(int)
    for s in steps:
        for k, v in s["building_started"].items():
            accum[k] += v
    base["building_started"] = dict(accum)
    base["lair_started"] = int(any(s["lair_started"] for s in steps))
    base["hive_started"] = int(any(s["hive_started"] for s in steps))

    return base


def extract_unit_details(unit: Unit):
    return {
        "tag": unit.tag,
        "last_seen_position": unit.position,
        "unit_type": unit.type_id,
        "is_structure": unit.is_structure,
        "armor": unit.armor,
        "movement_speed": unit.movement_speed,
        "health_max": unit.health_max,
        "health": unit.health,
        "shield_max": unit.shield_max,
        "shield": unit.shield,
        "energy": unit.energy,
        "energy_max": unit.energy_max,
    }


class _ObservationAggregator(ObserverAI):
    """
    Internal Class that abstracts away the Observer interface,
    use `ReplaySimulator` directly to interact and extract data
    from replays
    """

    def __init__(
        self,
        step_size: int,
        save_name: str,
        player_pov: int = 0,
        save_interval: int = 100,
    ):
        self._lair_started_this_step = None
        self._hive_started_this_step = None
        self.recent_steps = []
        self.iteration: int = 0
        self.step_size = step_size
        self.number_of_units = dict()
        self.visibility = np.ndarray

        self.enemy_units_seen_and_alive: Dict[int, Dict] = {}

        self.player_units: Dict[int, Dict] = {}
        self.prev_player_units = {}
        self.newly_queued = {unit.value: 0 for unit in VALID_UNITS[Race.Zerg]}
        self.building_started = {unit.value: 0 for unit in VALID_UNITS[Race.Zerg]}
        self._lair_started = 0
        self._hive_started = 0

        self.player_pov: int = player_pov
        self.workers_built = 0
        self.army_built = 0
        self.final_data = {}
        self.under_construction = {unit.value: 0 for unit in VALID_UNITS[Race.Zerg]}

        self.save_interval = save_interval
        self.save_name = save_name
        self.block_index = 0

    def _other(self, x: int = -1) -> int:
        if x == -1:
            return self._other(self.player_pov)
        if x == 1:
            return 2
        if x == 2:
            return 1
        return 0

    async def on_start(self):
        # Game engine advances step_size gameloops before sending observation
        print("POV:", self.player_pov)
        print("self.race:", self.race)
        print("enemy_race:", self.enemy_race)
        self.client.game_step = self.step_size
        if self.player_pov in (1, 2) and self.race != Race.Zerg:
            print(f"Skipping replay: POV {self.player_pov} is {self.race}")
            await self.client.leave()

    async def on_unit_created(self, unit: Unit):
        """Override this in your bot class. This function is called when a unit is created.

        :param unit:"""

    async def on_enemy_unit_entered_vision(self, unit: Unit) -> None:
        """
        Override this in your bot class. This function is called when an enemy unit (unit or structure) entered vision (which was not visible last frame).

        :param unit:
        """
        details = extract_unit_details(unit)
        details["last_seen"] = self.iteration
        self.enemy_units_seen_and_alive[unit.tag] = details

    async def on_unit_destroyed(self, unit_tag):
        """
        Override this in your bot class.
        Note that this function uses unit tags and not the unit objects
        because the unit does not exist anymore.

        :param unit_tag:
        """
        if self.enemy_units_seen_and_alive.get(unit_tag) is not None:
            del self.enemy_units_seen_and_alive[unit_tag]

        if self.player_units.get(unit_tag) is not None:
            del self.player_units[unit_tag]

    async def on_building_construction_started(self, unit: Unit):
        """
        Override this in your bot class.
        This function is called when a building construction has started.

        :param unit:
        """
        # Only track own buildings
        if unit.owner_id != self.player_pov:
            return

        # Only track buildings you care about
        if unit.type_id in VALID_UNITS[Race.Zerg]:
            self.building_started[unit.type_id.value] += 1

    async def on_unit_type_changed(self, unit: Unit, previous_type: UnitTypeId) -> None:
        """Override this in your bot class. This function is called when a unit type has changed. To get the current UnitTypeId of the unit, use 'unit.type_id'

        This may happen when a larva morphed to an egg, siege tank sieged, a zerg unit burrowed, a hatchery morphed to lair,
        a corruptor morphed to broodlordcocoon, etc..

        Examples::

            print(f"My unit changed type: {unit} from {previous_type} to {unit.type_id}")
        appears to only track units? Not structures? I need to make something else to track structures.
        :param unit:
        :param previous_type:
        """
        # Todo add tracking for queens
        # I believe i can use this for tracking if a queen is being produced. I think previous_type is hatchery.
        # newly_queued is used to track what type freshly enters queue
        if unit.orders:
            if (
                get_unit_type(unit.orders[0].ability.button_name)
                in VALID_UNITS[Race.Zerg]
                and previous_type in VALID_UNITS[Race.Zerg]
            ):
                self.newly_queued[
                    get_unit_type(unit.orders[0].ability.button_name).value
                ] += 1

        else:
            if unit.type_id == UnitTypeId.LAIR:
                print("hi")
            if unit.type_id in VALID_UNITS[Race.Zerg]:
                self.newly_queued[unit.type_id.value] += 1

    def already_pending_upgrade(self, upgrade_type: UpgradeId) -> float:
        """Check if an upgrade is being researched

        Returns values are::

            0 # not started
            0 < x < 1 # researching
            1 # completed

        Example::

            stim_completion_percentage = self.already_pending_upgrade(UpgradeId.STIMPACK)

        :param upgrade_type:
        """
        assert isinstance(upgrade_type, UpgradeId), f"{upgrade_type} is no UpgradeId"
        if upgrade_type in self.state.upgrades:
            return 1
        creationAbilityID = self.game_data.upgrades[
            upgrade_type.value
        ].research_ability.exact_id
        for structure in self.structures.filter(lambda unit: unit.is_ready):
            for order in structure.orders:
                if order.ability.exact_id == creationAbilityID:
                    return order.progress
        return 0

    def already_pending(self, unit_type: UpgradeId | UnitTypeId) -> float:
        """
        Returns a number of buildings or units already in progress, or if a
        worker is en route to build it. This also includes queued orders for
        workers and build queues of buildings.

        Example::

            amount_of_scv_in_production: int = self.already_pending(UnitTypeId.SCV)
            amount_of_CCs_in_queue_and_production: int = self.already_pending(UnitTypeId.COMMANDCENTER)
            amount_of_lairs_morphing: int = self.already_pending(UnitTypeId.LAIR)

        :param unit_type:
        """
        if isinstance(unit_type, UpgradeId):
            return self.already_pending_upgrade(unit_type)
        try:
            ability = self.game_data.units[unit_type.value].creation_ability.exact_id
        except AttributeError:
            if unit_type in CREATION_ABILITY_FIX:
                # Hotfix for checking pending archons
                if unit_type == UnitTypeId.ARCHON:
                    return (
                        self._abilities_count_and_build_progress[0][
                            AbilityId.ARCHON_WARP_TARGET
                        ]
                        / 2
                    )
                # Hotfix for rich geysers
                return self._abilities_count_and_build_progress[0][
                    CREATION_ABILITY_FIX[unit_type]
                ]
            return 0
        return self._abilities_count_and_build_progress[0][ability]

    async def on_end(self, game_result):
        """Flush any remaining steps that never filled a full block of 10."""
        if self.player_pov in (1, 2) and self.race != Race.Zerg:
            return
        if self.recent_steps:
            compressed = compress_step_block(self.recent_steps)
            self.final_data[self.block_index] = compressed
            self.block_index += 1
            self.recent_steps.clear()

        if self.final_data:
            save_number = (self.iteration // self.save_interval) + 1
            save_name = self.save_name + "_" + str(save_number)
            self.save_data(save_name)
            print(
                f"[on_end] Flushed {len(self.final_data)} block(s) to {save_name}.json.gz"
            )
        else:
            print("[on_end] Nothing to flush.")

    def save_data(self, output_name):
        data = self.final_data
        print(f"[SAVE] Writing {len(data)} blocks to {output_name}.json.gz")
        with gzip.open(output_name + ".json.gz", "wt", encoding="utf-8") as f:
            json.dump(data, f, cls=CustomEncoder)
            f.flush()
            gc.collect()

    def _prepare_step(self, state, proto_game_info):
        """
        :param state:
        :param proto_game_info:
        """

        # Set attributes from new state before on_step."""
        self.state: GameState = state  # See game_state.py
        # Required for events, needs to be before self.units are initialized so the old units are stored
        self._units_previous_map: Dict = {unit.tag: unit for unit in self.units}
        self._structures_previous_map: Dict = {
            structure.tag: structure for structure in self.structures
        }
        self.minerals: int = state.common.minerals
        self.vespene: int = state.common.vespene
        self.supply_army: int = state.common.food_army
        self.supply_workers: int = (
            state.common.food_workers
        )  # Doesn't include workers in production
        self.supply_cap: int = state.common.food_cap
        self.supply_used: int = state.common.food_used
        self.supply_left: int = self.supply_cap - self.supply_used
        self._prepare_units()

    async def on_step(self, iteration: int):
        self.iteration = iteration

        # Initialize the full-sized visibility map with -1
        self.visibility = np.full((256, 256), -1, dtype=int)

        x, y, w, h = self.game_info.playable_area
        playable_patch = self.state.visibility.data_numpy[y : y + h, x : x + w]

        # Decide where to place it in the 256×256 map.
        self.visibility[y : y + h, x : x + w] = playable_patch

        # Counts unit number
        self.number_of_units[iteration] = self.all_units.amount

        self.prev_player_units = self.player_units.copy()

        # Tracking what enemy units detected
        for unit in self.units:
            if unit.owner_id == self._other() and unit.is_visible:
                details = extract_unit_details(unit)
                details["last_seen"] = iteration
                self.enemy_units_seen_and_alive[unit.tag] = details
            # Tracking what new units are made + adding to army
            if unit.owner_id == self.player_pov and not unit.is_structure:
                if unit.tag not in self.player_units:
                    if unit.type_id in WORKERS:
                        self.workers_built += 1
                    # TODO i think update this to use supply instead. But i cant find where they keep supply cost for
                    if unit.type_id not in NOT_ARMY:
                        self.army_built += 1
                details = extract_unit_details(unit)
                self.player_units[unit.tag] = details

        x_off, y_off, w, h = self.game_info.playable_area
        # Scale minimap [0–64] → playable area [x_off:x_off+w, y_off:y_off+h]
        px = int(self.start_location.x / 64 * w) + x_off
        py = int(self.start_location.y / 64 * h) + y_off

        ex = int(self.enemy_start_locations[0].x / 64 * w) + x_off
        ey = int(self.enemy_start_locations[0].y / 64 * w) + y_off

        # make buildings and units that recently started construction, not ending construction
        for unit in VALID_UNITS[Race.Zerg]:
            id = unit.value
            self.under_construction[id] = int(self.already_pending(unit))

        # Todo Add tracking for greater spire + lurker den morphs
        # Lair and Hive tracking, possible to add greater spire/lurker den if necessary
        # One-shot detection for Lair
        if not self._lair_started:
            if self.already_pending(UnitTypeId.LAIR) > 0:
                self._lair_started = True
                self._lair_started_this_step = True
            else:
                self._lair_started_this_step = False
        else:
            self._lair_started_this_step = False

        # One-shot detection for Hive
        if not self._hive_started:
            if self.already_pending(UnitTypeId.HIVE) > 0:
                self._hive_started = True
                self._hive_started_this_step = True
            else:
                self._hive_started_this_step = False
        else:
            self._hive_started_this_step = False

        one_step: StepData = {
            "iteration": iteration,
            "visibility": self.visibility.tolist(),
            "enemy_units_seen_and_alive": self.enemy_units_seen_and_alive.copy(),
            "player_pov": self.player_pov,
            "player_units": self.player_units.copy(),
            "workers_built": self.workers_built,
            "army_built": self.army_built,
            "supply_cap": self.supply_cap,
            "supply_left": self.supply_left,
            "supply_used": self.supply_used,
            "supply_army": self.supply_army,
            "supply_workers": self.supply_workers,
            "own_race": self.race,
            "enemy_race": self.enemy_race,
            "own_spawn_x": px,
            "own_spawn_y": py,
            "enemy_spawn_x": ex,
            "enemy_spawn_y": ey,
            "minerals": self.minerals,
            "gas": self.vespene,
            "under_construction": self.under_construction.copy(),
            "newly_queued": self.newly_queued.copy(),
            "building_started": self.building_started.copy(),
            "lair_started": int(self._lair_started_this_step),
            "hive_started": int(self._hive_started_this_step),
        }

        self.recent_steps = getattr(self, "recent_steps", [])
        self.recent_steps.append(one_step)

        if len(self.recent_steps) == 10:
            compressed = compress_step_block(self.recent_steps)
            self.final_data[self.block_index] = compressed
            self.block_index += 1
            self.recent_steps.clear()

        if iteration != 0 and iteration % self.save_interval == 0:
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            summary.print_(sum1)
            save_number = iteration // self.save_interval
            save_name = self.save_name + "_" + str(save_number)
            self.save_data(save_name)

            if self.block_index > 0:
                first_step = self.final_data[self.block_index - 1]
                self.final_data.clear()
                self.final_data[0] = first_step
                self.block_index = 1
            else:
                # Nothing to preserve, just clear
                self.final_data.clear()
                self.block_index = 0

            self.visibility = []
            print("data_cleared")
            if not self.iteration == 0:
                print("Memory usage by component:")
                print("player_units:", asizeof.asizeof(self.player_units))
                print(
                    "enemy_units_seen_and_alive:",
                    asizeof.asizeof(self.enemy_units_seen_and_alive),
                )
                print("recent_steps:", asizeof.asizeof(self.recent_steps))
                print("final_data:", asizeof.asizeof(self.final_data))
                print("visibility:", asizeof.asizeof(self.visibility))
            if self.time > 1200.0:
                print("[FORCE END] 20 minute cutoff reached")
                await self.client.leave()
                return
        print_memory_usage()
        self.newly_queued = {unit.value: 0 for unit in VALID_UNITS[Race.Zerg]}
        self.under_construction = {unit.value: 0 for unit in VALID_UNITS[Race.Zerg]}
        self.workers_built = 0
        self.army_built = 0
        self.building_started = {unit.value: 0 for unit in VALID_UNITS[Race.Zerg]}


class ReplaySimulator:
    """
    Each `ReplaySimulator` is bound to exactly one replay path, and `run_simulation()` must be called
    before any additional data can be extracted from an instance of `ReplaySimulator`

    Example Usage::

        path = "tests/replays/Ultralove.SC2Replay"
        simulator = ReplaySimulator(path, step_size=60, fow_pov=2)
        simulator.run_simulation()
        lifetimes = simulator.get_unit_lifetimes()
        print(lifetimes)
    """

    def __init__(
        self, path: str, save_name: str, step_size: int = 10, fow_pov: int = 0
    ):
        """
        :param path: Relative or absolute path of replay
        :param step_size: Number of gameloops to skip before collecting observation data.
            Increase step size if performance is too slow.
            For reference, 22.4 gameloops happen every second
        :param fow_pov: Perspective from which fog of war should be observed from. Set this
            to 0 to disable fog of war, 1 to spectate from Player 1's POV, and 2 for Player 2's POV
        """
        replay_path = self._validate_path(path)
        self.replay_path = replay_path
        self.observer = _ObservationAggregator(
            step_size, save_name=save_name, player_pov=fow_pov
        )
        self.completed_simulation = False
        self.fow_pov = fow_pov

    def _validate_path(self, path: str) -> str:
        replay_name = path
        if platform.system() == "Linux":
            home_replay_folder = Path.home() / "Documents" / "StarCraft II" / "Replays"
            replay_path = home_replay_folder / replay_name
            if not replay_path.is_file():
                raise FileNotFoundError
            replay_path = str(replay_path)
        elif os.path.isabs(replay_name):
            replay_path = replay_name
        else:
            # Convert relative path to absolute path, assuming this replay is in this folder
            folder_path = os.path.dirname(__file__)
            replay_path = os.path.join(folder_path, replay_name)
        assert os.path.isfile(replay_path), f"Replay not found: {replay_path}"
        return replay_path

    def run_simulation(self) -> None:
        """
        This function must be called before any other getter functions can be used
        """
        run_replay(
            self.observer,
            replay_path=self.replay_path,
            realtime=False,
            observed_id=self.fow_pov,
        )
        self.completed_simulation = True

    def get_visibility_map(self):
        assert (
            self.completed_simulation
        ), "Call simulator.run_simulation() before using this function!"
        return self.observer.visibility

    def get_unit_counts(self):
        assert (
            self.completed_simulation
        ), "Call simulator.run_simulation() before using this function!"
        return self.observer.number_of_units

    def get_final_data(self):
        assert (
            self.completed_simulation
        ), "Call simulator.run_simulation() before using this function!"
        return self.observer.final_data


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UnitTypeId):
            return obj.value
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Race):
            return (
                2
                # zerg
                if obj == Race.Zerg
                else (
                    1
                    # terran
                    if obj == Race.Terran
                    else (
                        3
                        # protoss
                        if obj == Race.Protoss
                        else 4 if obj == Race.Random else 5
                        # random, unknown
                    )
                )
            )
        return super().default(obj)


# e.g. replay_name = "tests/replays/Alcyone LE (6).SC2Replay"
# e.g. output_name = "output.json.gz"
# 224 step size is 10s
def extract_data(replay_name: str, output_name: str, fow_pov, step_size: int = 20):
    simulator = ReplaySimulator(
        replay_name, output_name, fow_pov=fow_pov, step_size=step_size
    )
    simulator.run_simulation()
    data = simulator.get_final_data()
    return data


if __name__ == "__main__":
    # print("hi")
    # simulator = ReplaySimulator("1000 replays/26382815.SC2Replay", fow_pov=1, step_size=224)
    # simulator.run_simulation()
    # pixelmap_x_length, pixelmap_y_length = simulator.observer.state.visibility.data_numpy.shape
    # extract_data("tests/replays/Alcyone LE (7).SC2Replay", "output.json.gz", 1)

    if len(sys.argv) == 5:
        assert (
            len(sys.argv) == 5
        ), "Usage: python simulator.py replay_path output_path fow_pov step_size"
        replay_path = sys.argv[1]
        output_path = sys.argv[2]
        fow_pov = int(sys.argv[3])
        step_size = int(sys.argv[4])
        extract_data(replay_path, output_path, fow_pov, step_size)
    else:

        replays = [
            "noscoutnoresponse.SC2Replay",
            "scoutbadresponse.SC2Replay",
            "scoutnoresponse.SC2Replay",
            "scoutlessbadresponse.SC2Replay",
        ]
        for replay in replays:
            path = "1000 replays/" + replay
            save_name = "1000 extracts/" + replay
            simulator = ReplaySimulator(
                # "tests/replays/Alcyone LE (6).SC2Replay",
                path,
                # "1000 replays/10000 Feet LE (3).SC2Replay",
                save_name=save_name,
                fow_pov=1,
                step_size=20,
            )
            simulator.run_simulation()
