import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

from sc2.game_state import GameState
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_replay
from sc2.observer_ai import ObserverAI

from models import UnitLifetime, UnitPosition
from sc2.unit import Unit

import gzip

MAX_GRID_SIZE = 182
NUMBER_OF_GRIDS = 20
SIZE_OF_GRID = MAX_GRID_SIZE/NUMBER_OF_GRIDS
NUM_PREDICTED_STEPS = 2
WORKERS = [UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE]
NOT_ARMY = WORKERS + [UnitTypeId.OVERLORD, UnitTypeId.OVERSEER, UnitTypeId.OVERLORDTRANSPORT]

def extract_unit_details(unit: Unit):
     return {"tag": unit.tag,
            "UNITTYPEID": unit.type_id,
             "last_seen_position": unit.position,
             "unit_type": unit.type_id,
             "is_structure": unit.is_structure,
             "is_light": unit.is_light,
             "is_armored": unit.is_armored,
             "is_biological": unit.is_biological,
             "is_technical": unit.is_mechanical,
             "is_massive": unit.is_massive,
             "is_psionic": unit.is_psionic,
             "can_attack": unit.can_attack,
             "can_attack_both": unit.can_attack_both,
             "can_attack_ground": unit.can_attack_ground,
             "ground_dps": unit.ground_dps,
             "ground_range": unit.ground_range,
             "can_attack_air": unit.can_attack_air,
             "air_dps": unit.air_dps,
             "air_range": unit.air_range,
             "bonus_damage": unit.bonus_damage,
             "armor": unit.armor,
             "movement_speed": unit.movement_speed,
             "real_speed": unit.real_speed,
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

    def __init__(self, step_size: int, player_pov = 0):
        self.prev_player_buildings = {}
        self.step_size = step_size
        self.lifetimes = dict()
        self.visibility = np.ndarray
        self.number_of_units = dict()
        self.enemy_units_seen_and_alive = {}
        self.player_pov = player_pov
        self.buildings_constructed = {0: [], 1: [], 2: []}
        self.new_buildings = []
        self.player_buildings = {}
        self.player_army = {}
        self.new_units = []
        self.units_built = {0: [], 1: [], 2: []}
        self.workers_built = 0
        self.army_built = 0
        self.prev_player_units = {}
        #These are supply things already done
        # self.supply_cap
        # self.supply_left
        # self.supply_used
        # self.supply_army
        # self.supply_army
        self.data = {}
        self.units2 = {}

    def _other(self, x:int = -1) -> int:
        if x == -1:
            return self._other(self.player_pov)
        if x == 1:
            return 2
        if x == 2:
            return 1
        return 0

    async def on_start(self):
        # Game engine advances step_size gameloops before sending observation
        self.client.game_step = self.step_size

    async def on_unit_created(self, unit: Unit):
        """Override this in your bot class. This function is called when a unit is created.

        :param unit:"""
        if unit.is_structure and unit.owner_id == self.player_pov:
            self.new_buildings.append(unit)
        if not unit.is_structure and unit.owner_id == self.player_pov:
            self.new_units.append(unit)

    async def on_unit_destroyed(self, unit_tag):
        """
        Override this in your bot class.
        Note that this function uses unit tags and not the unit objects
        because the unit does not exist anymore.

        :param unit_tag:
        """
        if self.enemy_units_seen_and_alive.get(unit_tag) is not None:
            del self.enemy_units_seen_and_alive[unit_tag]

        if self.player_army.get(unit_tag) is not None:
            del self.player_army[unit_tag]

        if self.player_buildings.get(unit_tag) is not None:
            del self.player_buildings[unit_tag]

    #testing to see if overriding breaks anything
    #ok it doesnt. observer ai overrides this supposedly final method with some other garbage that broke support for
    #supply_army and supply_workers. I put them back so i can use them for tracking total worker count and total
    #army count
    def _prepare_step(self, state, proto_game_info):
        """
        :param state:
        :param proto_game_info:
        """

        # Set attributes from new state before on_step."""
        self.state: GameState = state  # See game_state.py
        # Required for events, needs to be before self.units are initialized so the old units are stored
        self._units_previous_map: Dict = {unit.tag: unit for unit in self.units}
        self._structures_previous_map: Dict = {structure.tag: structure for structure in self.structures}
        self.supply_army: int = state.common.food_army
        self.supply_workers: int = state.common.food_workers
        self.minerals: int = state.common.minerals
        self.vespene: int = state.common.vespene
        self.supply_army: int = state.common.food_army
        self.supply_workers: int = state.common.food_workers  # Doesn't include workers in production
        self.supply_cap: int = state.common.food_cap
        self.supply_used: int = state.common.food_used
        self.supply_left: int = self.supply_cap - self.supply_used
        self._prepare_units()

    #todo include supply cap + current max supply
    async def on_step(self, iteration: int):
        # TODO: Only basic information is included for now, need to add more
        # stuff to aggregate later on

        self.new_buildings = []
        self.new_units = []
        self.workers_built = 0
        self.army_built = 0
        # Add Unit lifetime data
        for i in range(len(self.units)):
            unit = self.units[i]
            position = UnitPosition(unit.game_loop, unit.position[0], unit.position[1])
            if unit.tag not in self.lifetimes:
                self.lifetimes[unit.tag] = UnitLifetime(
                    unit.tag, unit.owner_id, unit.name, [position], False, -1
                )
            else:
                self.lifetimes[unit.tag].positions.append(position)


        # Initialize the full-sized visibility map with -1
        self.visibility = np.full((248, 248), -1, dtype=int)

        # Your 176×176 visibility input
        # visibility = np.array(...)

        # Playable region inside the 176×176 visibility array
        x, y, w, h = self.game_info.playable_area
        playable_patch = self.state.visibility.data_numpy[y:y + h, x:x + w]  # (140×140)

        # Decide where to place it in the 248×248 map.
        # We'll center it at same position: starting at (18, 16)
        self.visibility[y:y + h, x:x + w] = playable_patch


        # Counts unit number
        self.number_of_units[iteration] = self.all_units.amount

        self.prev_player_buildings = self.player_buildings.copy()

        self.prev_player_units = self.player_army.copy()

        #Tracking what enemy units detected
        for unit in self.units:
            if unit.owner_id == self._other() and unit.is_visible:
                details = extract_unit_details(unit)
                details["last_seen"] = iteration
                self.enemy_units_seen_and_alive[unit.tag] = details
                self.units2[unit.tag] = unit
            #Tracking what new units are made + adding to army
            if unit.owner_id == self.player_pov and not unit.is_structure:
                if unit.tag not in self.player_army:
                    self.new_units.append(unit)
                    self.units2[unit.tag] = unit
                    if unit.type_id in WORKERS:
                        self.workers_built += 1
                    #TODO i think update this to use supply instead. But i cant find where they keep supply cost for
                    # unit

                    if unit.type_id not in NOT_ARMY:
                        self.army_built += 1

                details = extract_unit_details(unit)
                self.player_army[unit.tag] = details
                self.units2[unit.tag] = unit

            #tracking total structures + new structures
            if unit.owner_id == self.player_pov and unit.is_structure:
                if unit.tag not in self.player_buildings:
                    self.new_buildings.append(unit)
                    self.units2[unit.tag] = unit
                self.player_buildings[unit.tag] = unit
                self.units2[unit.tag] = unit


        #Tracking unit morphs
        for tag in self.player_army:
            unit = self.units2[tag]
            if tag in self.prev_player_units and self.prev_player_units[tag]["UNITTYPEID"] != unit.type_id:
                self.new_units.append(unit)
                self.units2[unit.tag] = unit

        #tracking building morphs
        for tag in self.player_buildings:
            building = self.player_buildings[tag]
            if tag in self.prev_player_buildings and self.prev_player_buildings[tag].name != building.name:
                self.new_buildings.append(building)
                self.units2[building.tag] = building


        self.buildings_constructed[2] = self.buildings_constructed[1].copy()
        self.buildings_constructed[1] = self.buildings_constructed[0].copy()
        self.buildings_constructed[0] = self.new_buildings

        self.units_built[2] = self.units_built[1]
        self.units_built[1] = self.units_built[0]
        self.units_built[0] = self.new_units

        x_off, y_off, w, h = self.game_info.playable_area
        # Scale minimap [0–64] → playable area [x_off:x_off+w, y_off:y_off+h]
        px = int(self.start_location.x / 64 * w) + x_off
        py = int(self.start_location.y / 64 * h) + y_off

        ex = int(self.enemy_start_locations[0].x / 64 * w) + x_off
        ey = int(self.enemy_start_locations[0].y / 64 * w) + x_off


        self.data[iteration] = {
            "iteration": iteration,
            "visibility": self.visibility,
            "enemy_units_seen_and_alive": self.enemy_units_seen_and_alive,
            "player_pov": self.player_pov,
            "buildings_constructed": self.buildings_constructed,
            "new_buildings": self.new_buildings,
            "player_buildings": self.player_buildings,
            "player_army": self.player_army,
            "new_units": self.new_units,
            "units_built": self.units_built,
            "workers_built": self.workers_built,
            "army_built": self.army_built,
            "prev_player_units": self.prev_player_units,
            "supply_cap": self.supply_cap,
            "supply_left": self.supply_left,
            "supply_used": self.supply_used,
            "supply_army": self.supply_army,
            "supply_workers": self.supply_workers,
            "own_race":self.race,
            "enemy_race":self.enemy_race,
            "own_spawn_x":px,
            "own_spawn_y":py,
            "enemy_spawn_x":ex,
            "enemy_spawn_y":ey
                                }
        # self.supply_cap
        # self.supply_left
        # self.supply_used
        # self.supply_army
        # self.supply_workers
        # self.prev_player_buildings = {}
        # self.step_size = step_size
        # self.visibility = []
        # self.enemy_units_seen_and_alive = {}
        # self.player_pov = player_pov
        # self.buildings_constructed = {0: [], 1: [], 2: []}
        # self.new_buildings = []
        # self.player_buildings = {}
        # self.player_army = {}
        # self.new_units = []
        # self.units_built = {0: [], 1: [], 2: []}
        # self.workers_built = 0
        # self.army_built = 0
        # self.prev_player_units = {}



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

    def __init__(self, path: str, step_size: int = 22, fow_pov: int = 0):
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
        self.observer = _ObservationAggregator(step_size, fow_pov)
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

    def get_unit_lifetimes(self) -> List[UnitLifetime]:
        assert (
            self.completed_simulation
        ), "Call simulator.run_simulation() before using this function!"
        return self.observer.lifetimes

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

    def get_data(self):
        assert (
            self.completed_simulation
        ), "Call simulator.run_simulation() before using this function!"
        return self.observer.data

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UnitTypeId):
            return obj.value
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Unit):
            return extract_unit_details(obj)
        return super().default(obj)

#e.g. replay_name = "tests/replays/Alcyone LE (3).SC2Replay"
#e.g. output_name = "output.json.gz"
#224 step size is 10s
def extract_data(replay_name: str, output_name: str, fow_pov, step_size: int = 224):
    simulator = ReplaySimulator(replay_name, fow_pov=fow_pov, step_size=step_size)
    simulator.run_simulation()
    data = simulator.get_data()


    with gzip.open(output_name, "wt", encoding="utf-8") as f:
        json.dump(data, f, cls=CustomEncoder)

    return data

def process_folder(input_folder="1000 replays", output_folder="1000 extracts"):
    t0 = time.time()
    count = 0
    os.makedirs(output_folder, exist_ok=True)
    folder_path = input_folder
    print("count number: " + str(count))
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)
        output_path_p1 = output_path + "_p1.json.gz"
        output_path_p2 = output_path + "_p2.json.gz"

        if os.path.exists(output_path_p1) and os.path.exists(output_path_p2):
            print("skipped")
            continue
        if os.path.isfile(file_path):
            extract_data(file_path, output_name=output_path + "_p1.json.gz", fow_pov=1)
            extract_data(file_path, output_name=output_path + "_p2.json.gz", fow_pov=2)
        elapsed = time.time() - t0
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        print(f"{hours}h {minutes}m {seconds}s")
        print(datetime.now().strftime("%H:%M:%S"))  # 24-hour time

if __name__ == "__main__":
    print("hi")
    simulator = ReplaySimulator("tests/replays/Amphion LE.SC2Replay", fow_pov=1, step_size=22)
    simulator.run_simulation()
    pixelmap_x_length, pixelmap_y_length = simulator.observer.state.visibility.data_numpy.shape




