import os
import sys
import platform
from contextlib import nullcontext
from pathlib import Path
from typing import List

import sc2.units
from sc2.main import run_replay
from sc2.observer_ai import ObserverAI

from models import UnitLifetime, UnitPosition
from sc2.unit import Unit

MAX_GRID_SIZE = 182
NUMBER_OF_GRIDS = 20
SIZE_OF_GRID = MAX_GRID_SIZE/NUMBER_OF_GRIDS
NUM_PREDICTED_STEPS = 2



class _ObservationAggregator(ObserverAI):
    """
    Internal Class that abstracts away the Observer interface,
    use `ReplaySimulator` directly to interact and extract data
    from replays
    """

    def __init__(self, step_size: int, player_pow = 0):
        self.step_size = step_size
        self.lifetimes = dict()
        self.visibility = []
        self.number_of_units = dict()
        self.follow_unit = None
        self.enemy_units_seen_and_alive = {}
        self.player_pov = player_pow
        self.player_actions = {}
        self.buildings_constructed = {0: [], 1: [], 2: []}
        self.new_buildings = []

    def _other(self, x: int) -> int:
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

    async def on_step(self, iteration: int):
        # TODO: Only basic information is included for now, need to add more
        # stuff to aggregate later on

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

        # Add player visibility data
        self.visibility.append(self.state.visibility.data_numpy)

        # Counts unit number
        self.number_of_units[iteration] = self.all_units.amount



        player_units = []
        for unit in self.units:
            if unit.owner_id == self._other(self.player_pov) and unit.is_visible:
                player_units.append(unit)
                self.enemy_units_seen_and_alive[unit.tag] = {"unit": unit,
                                                             "last_seen_position": unit.position,
                                                             "unit_type": unit.type_id,
                                                             "last_seen": iteration,
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
                                                             "shield_max": unit.shield,
                                                             "shield": unit.shield,
                                                             "energy": unit.energy,
                                                             "energy_max": unit.energy_max,
                                                             }


        #print(self.enemy_units_seen_and_alive)
        print(iteration)
        #print(player_units)
        self.buildings_constructed[2] = self.buildings_constructed[1].copy()
        self.buildings_constructed[1] = self.buildings_constructed[0].copy()
        self.buildings_constructed[0] = self.new_buildings
        print("new buildings:")
        print(self.new_buildings)
        self.new_buildings = []




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


# Example use of the ReplaySimulator
path = "tests/replays/2025-06-03 - (Z)ISay VS (T)kai.SC2Replay"
simulator = ReplaySimulator(path, fow_pov=1)
#, step_size=60
simulator.run_simulation()
visibility = simulator.get_visibility_map()