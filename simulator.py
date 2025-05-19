import os
import sys
import platform
from pathlib import Path
from typing import List

from sc2.main import run_replay
from sc2.observer_ai import ObserverAI

from models import UnitLifetime, UnitPosition


class _ObservationAggregator(ObserverAI):

    def __init__(self, step_size: int):
        self.lifetimes = dict()
        self.step_size = step_size

    async def on_start(self):
        # Game engine advances step_size gameloops before sending observation
        self.client.game_step = self.step_size

    async def on_step(self, iteration: int):
        # TODO: Only basic information is included for now, need to add more
        # stuff to aggregate later on
        for i in range(len(self.units)):
            unit = self.units[i]
            position = UnitPosition(unit.game_loop, unit.position[0], unit.position[1])
            if unit.tag not in self.lifetimes:
                self.lifetimes[unit.tag] = UnitLifetime(
                    unit.tag, unit.owner_id, unit.name, [position], False, -1
                )
            else:
                self.lifetimes[unit.tag].positions.append(position)


class ReplaySimulator:
    def __init__(self, path: str, step_size: int):
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

        self.replay_path = replay_path
        self.observer = _ObservationAggregator(step_size)

    def run_simulation(self) -> List[UnitLifetime]:
        run_replay(
            self.observer, replay_path=self.replay_path, realtime=False, observed_id=1
        )
        return self.observer.lifetimes


path = "tests/replays/Ultralove.SC2Replay"
simulator = ReplaySimulator(path, step_size=60)
lifetimes = simulator.run_simulation()
print(lifetimes)
