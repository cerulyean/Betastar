from dataclasses import dataclass


@dataclass
class Message:
    gameloop: int
    sender: int
    message: str


@dataclass
class BuildingEvent:
    gameloop: int
    buildingName: str
    player: int
    x: int
    y: int
    isFinished: bool  # False when building starts construction, True when completed
    buildingTag: int


@dataclass
class MacroStatistics:
    gameloop: int
    player: int
    minerals: int
    vespene: int
    mineralIncome: int
    vespeneIncome: int
    workers: int
    mineralUsedInProgressArmy: int
    mineralUsedInProgressEconomy: int
    mineralUsedInProgressTechnology: int
    vespeneUsedInProgressArmy: int
    vespeneUsedInProgressEconomy: int
    vespeneUsedInProgressTechnology: int
    mineralArmyValue: int
    mineralEconomyValue: int
    mineralTechnologyValue: int
    vespeneArmyValue: int
    vespeneEconomyValue: int
    vespeneTechnologyValue: int
    mineralsLostArmy: int
    mineralsLostEconomy: int
    mineralsLostTechnology: int
    vespeneLostArmy: int
    vespeneLostEconomy: int
    vespeneLostTechnology: int
    mineralsKilledArmy: int
    mineralsKilledEconomy: int
    mineralsKilledTechnology: int
    vespeneKilledArmy: int
    vespeneKilledEconomy: int
    vespeneKilledTechnology: int
