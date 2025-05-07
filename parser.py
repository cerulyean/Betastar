from mpyq import MPQArchive
from s2protocol import versions
from models import BuildingEvent, Message, MacroStatistics
from typing import List

# List of Non-Building Units that can emit SUnitInitEvents
UnitInitExclusionList = [
    b"Zealot",
    b"Sentry",
    b"Stalker",
    b"Adept",
    b"HighTemplar",
    b"DarkTemplar",
    b"Archon",
]


class Parser:
    def __init__(self, path: str) -> None:
        self.archive = MPQArchive(path)

        # Get replay version number and build protocol decoder for this version
        contents = self.archive.header["user_data_header"]["content"]
        header = versions.latest().decode_replay_header(contents)
        baseBuild = header["m_version"]["m_baseBuild"]
        self.protocol = versions.build(baseBuild)

        # Read and decode tracker events
        trackerContents = self.archive.read_file("replay.tracker.events")
        trackerEventGenerator = self.protocol.decode_replay_tracker_events(
            trackerContents
        )

        self.trackerEvents = []
        for event in trackerEventGenerator:
            self.trackerEvents.append(event)

    def get_messages(self) -> List[Message]:
        # Read and decode replay message events
        messageContents = self.archive.read_file("replay.message.events")
        messageEvents = self.protocol.decode_replay_message_events(messageContents)
        msgs = []
        for msg in messageEvents:
            if msg["_event"] == "NNet.Game.SChatMessage":
                sender = msg["_userid"]["m_userId"]
                gameloop = msg["_gameloop"]
                text = msg["m_string"]
                msgs.append(Message(sender, gameloop, text))
        return msgs

    def get_building_events(self) -> List[BuildingEvent]:
        # Extract Building Events
        tagRef = dict()
        buildingEvents = []
        for event in self.trackerEvents:
            if (
                event.get("m_unitTypeName") is not None
                and event["m_unitTypeName"] in UnitInitExclusionList
            ):
                continue

            if event["_event"] == "NNet.Replay.Tracker.SUnitInitEvent":
                name = event["m_unitTypeName"]
                gameloop = event["_gameloop"]
                player = event["m_controlPlayerId"]
                x = event["m_x"]
                y = event["m_y"]
                buildingTag = event["m_unitTagIndex"]
                isFinished = False
                bEvent = BuildingEvent(
                    gameloop, name, player, x, y, isFinished, buildingTag
                )
                buildingEvents.append(bEvent)
                tagRef[buildingTag] = bEvent
            elif event["_event"] == "NNet.Replay.Tracker.SUnitDoneEvent":
                targetIdx = event["m_unitTagIndex"]

                # Targeted unit does not exist or is not a building
                if tagRef.get(targetIdx) is None:
                    continue

                # Add building completed event
                t = tagRef[targetIdx]
                bEvent = BuildingEvent(
                    event["_gameloop"], t.buildingName, t.player, t.x, t.y, True, targetIdx
                )
                buildingEvents.append(bEvent)
        
        return buildingEvents

    def get_macro_statistics(self) -> List[MacroStatistics]:
        macroStats = []
        for e in self.trackerEvents:
            if e["_event"] == "NNet.Replay.Tracker.SPlayerStatsEvent":
                event = e["m_stats"]
                gameloop = e["_gameloop"]
                player = e["m_playerId"]
                minerals  = event["m_scoreValueMineralsCurrent"]
                vespene   = event["m_scoreValueVespeneCurrent"]
                mineralIncome  = event["m_scoreValueMineralsCollectionRate"]
                vespeneIncome  = event["m_scoreValueVespeneCollectionRate"]
                workers   = event["m_scoreValueWorkersActiveCount"]

                mineralUsedInProgressArmy       = event["m_scoreValueMineralsUsedInProgressArmy"]
                mineralUsedInProgressEconomy    = event["m_scoreValueMineralsUsedInProgressEconomy"]
                mineralUsedInProgressTechnology = event["m_scoreValueMineralsUsedInProgressTechnology"]
                vespeneUsedInProgressArmy       = event["m_scoreValueVespeneUsedInProgressArmy"]
                vespeneUsedInProgressEconomy    = event["m_scoreValueVespeneUsedInProgressEconomy"]
                vespeneUsedInProgressTechnology = event["m_scoreValueVespeneUsedInProgressTechnology"]

                mineralArmyValue       = event["m_scoreValueMineralsUsedCurrentArmy"]
                mineralEconomyValue    = event["m_scoreValueMineralsUsedCurrentEconomy"]
                mineralTechnologyValue = event["m_scoreValueMineralsUsedCurrentTechnology"]
                vespeneArmyValue       = event["m_scoreValueVespeneUsedCurrentArmy"]
                vespeneEconomyValue    = event["m_scoreValueVespeneUsedCurrentEconomy"]
                vespeneTechnologyValue = event["m_scoreValueVespeneUsedCurrentTechnology"]

                mineralsLostArmy       = event["m_scoreValueMineralsLostArmy"]
                mineralsLostEconomy    = event["m_scoreValueMineralsLostEconomy"]
                mineralsLostTechnology = event["m_scoreValueMineralsLostTechnology"]
                vespeneLostArmy        = event["m_scoreValueVespeneLostArmy"]
                vespeneLostEconomy     = event["m_scoreValueVespeneLostEconomy"]
                vespeneLostTechnology  = event["m_scoreValueVespeneLostTechnology"]

                mineralsKilledArmy       = event["m_scoreValueMineralsKilledArmy"]
                mineralsKilledEconomy    = event["m_scoreValueMineralsKilledEconomy"]
                mineralsKilledTechnology = event["m_scoreValueMineralsKilledTechnology"]
                vespeneKilledArmy        = event["m_scoreValueVespeneKilledArmy"]
                vespeneKilledEconomy     = event["m_scoreValueVespeneKilledEconomy"]
                vespeneKilledTechnology  = event["m_scoreValueVespeneKilledTechnology"]

                # --- build the MacroStatistics object, add to list ---
                macroStats.append(
                    MacroStatistics(
                        gameloop,
                        player,
                        minerals,
                        vespene,
                        mineralIncome,
                        vespeneIncome,
                        workers,
                        mineralUsedInProgressArmy,
                        mineralUsedInProgressEconomy,
                        mineralUsedInProgressTechnology,
                        vespeneUsedInProgressArmy,
                        vespeneUsedInProgressEconomy,
                        vespeneUsedInProgressTechnology,
                        mineralArmyValue,
                        mineralEconomyValue,
                        mineralTechnologyValue,
                        vespeneArmyValue,
                        vespeneEconomyValue,
                        vespeneTechnologyValue,
                        mineralsLostArmy,
                        mineralsLostEconomy,
                        mineralsLostTechnology,
                        vespeneLostArmy,
                        vespeneLostEconomy,
                        vespeneLostTechnology,
                        mineralsKilledArmy,
                        mineralsKilledEconomy,
                        mineralsKilledTechnology,
                        vespeneKilledArmy,
                        vespeneKilledEconomy,
                        vespeneKilledTechnology,
                    )
                )

        return macroStats

parser = Parser("tests/replays/AbyssalReefLE.SC2Replay")

buildingEvents = parser.get_macro_statistics()

for event in buildingEvents:
    print(event)