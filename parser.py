from mpyq import MPQArchive
from s2protocol import versions
from models import BuildingEvent, Message, MacroStatistics, UnitLifetime, UnitPosition
from utils import unit_tag
from typing import List
from collections import defaultdict

# List of Non-Building Units that can emit SUnitInitEvents
UnitInitExclusionList = set(
    [
        b"Zealot",
        b"Sentry",
        b"Stalker",
        b"Adept",
        b"HighTemplar",
        b"DarkTemplar",
        b"Archon",
    ]
)


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

    def __get_tracker_events(self):
        return self.trackerEvents

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
        tagRef = dict()  # tagRef uses globally unique building tag
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
                buildingTag = unit_tag(
                    event["m_unitTagIndex"], event["m_unitTagRecycle"]
                )
                isFinished = False
                bEvent = BuildingEvent(
                    gameloop, name, player, x, y, isFinished, buildingTag, False
                )
                buildingEvents.append(bEvent)
                tagRef[buildingTag] = bEvent
            elif event["_event"] == "NNet.Replay.Tracker.SUnitDoneEvent":
                targetIdx = unit_tag(event["m_unitTagIndex"], event["m_unitTagRecycle"])

                # Targeted unit does not exist or is not a building
                if tagRef.get(targetIdx) is None:
                    continue

                # Add building completed event
                t = tagRef[targetIdx]
                bEvent = BuildingEvent(
                    event["_gameloop"],
                    t.buildingName,
                    t.player,
                    t.x,
                    t.y,
                    True,
                    targetIdx,
                    False,
                )
                buildingEvents.append(bEvent)

            elif event["_event"] == "NNet.Replay.Tracker.SUnitDiedEvent":
                tag = unit_tag(event["m_unitTagIndex"], event["m_unitTagRecycle"])

                if tagRef.get(tag) is not None:
                    b = tagRef[tag]
                    gameloop = event["_gameloop"]
                    x = event["m_x"]
                    y = event["m_y"]
                    deadEvent = BuildingEvent(
                        gameloop,
                        b.buildingName,
                        b.player,
                        x,
                        y,
                        b.isFinished,
                        tag,
                        True,
                    )
                    buildingEvents.append(deadEvent)

        return buildingEvents

    def get_macro_statistics(self) -> List[MacroStatistics]:
        macroStats = []
        for e in self.trackerEvents:
            if e["_event"] == "NNet.Replay.Tracker.SPlayerStatsEvent":
                event = e["m_stats"]
                gameloop = e["_gameloop"]
                player = e["m_playerId"]
                minerals = event["m_scoreValueMineralsCurrent"]
                vespene = event["m_scoreValueVespeneCurrent"]
                mineralIncome = event["m_scoreValueMineralsCollectionRate"]
                vespeneIncome = event["m_scoreValueVespeneCollectionRate"]
                workers = event["m_scoreValueWorkersActiveCount"]

                mineralUsedInProgressArmy = event[
                    "m_scoreValueMineralsUsedInProgressArmy"
                ]
                mineralUsedInProgressEconomy = event[
                    "m_scoreValueMineralsUsedInProgressEconomy"
                ]
                mineralUsedInProgressTechnology = event[
                    "m_scoreValueMineralsUsedInProgressTechnology"
                ]
                vespeneUsedInProgressArmy = event[
                    "m_scoreValueVespeneUsedInProgressArmy"
                ]
                vespeneUsedInProgressEconomy = event[
                    "m_scoreValueVespeneUsedInProgressEconomy"
                ]
                vespeneUsedInProgressTechnology = event[
                    "m_scoreValueVespeneUsedInProgressTechnology"
                ]

                mineralArmyValue = event["m_scoreValueMineralsUsedCurrentArmy"]
                mineralEconomyValue = event["m_scoreValueMineralsUsedCurrentEconomy"]
                mineralTechnologyValue = event[
                    "m_scoreValueMineralsUsedCurrentTechnology"
                ]
                vespeneArmyValue = event["m_scoreValueVespeneUsedCurrentArmy"]
                vespeneEconomyValue = event["m_scoreValueVespeneUsedCurrentEconomy"]
                vespeneTechnologyValue = event[
                    "m_scoreValueVespeneUsedCurrentTechnology"
                ]

                mineralsLostArmy = event["m_scoreValueMineralsLostArmy"]
                mineralsLostEconomy = event["m_scoreValueMineralsLostEconomy"]
                mineralsLostTechnology = event["m_scoreValueMineralsLostTechnology"]
                vespeneLostArmy = event["m_scoreValueVespeneLostArmy"]
                vespeneLostEconomy = event["m_scoreValueVespeneLostEconomy"]
                vespeneLostTechnology = event["m_scoreValueVespeneLostTechnology"]

                mineralsKilledArmy = event["m_scoreValueMineralsKilledArmy"]
                mineralsKilledEconomy = event["m_scoreValueMineralsKilledEconomy"]
                mineralsKilledTechnology = event["m_scoreValueMineralsKilledTechnology"]
                vespeneKilledArmy = event["m_scoreValueVespeneKilledArmy"]
                vespeneKilledEconomy = event["m_scoreValueVespeneKilledEconomy"]
                vespeneKilledTechnology = event["m_scoreValueVespeneKilledTechnology"]

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

    def get_unit_lifetimes(self) -> List[UnitLifetime]:
        units = dict()  # Only stores units that are alive
        pastIndexHolders = defaultdict(lambda: [])
        deadUnits = []
        for event in self.trackerEvents:
            if event["_event"] == "NNet.Replay.Tracker.SUnitBornEvent" or (
                event["_event"] == "NNet.Replay.Tracker.SUnitInitEvent"
                and event["m_unitTypeName"] in UnitInitExclusionList
            ):
                # Starting Mineral and Vespene are all controlled by 0
                if event["m_controlPlayerId"] == 0:
                    continue
                tagIndex = event["m_unitTagIndex"]
                tagRecycle = event["m_unitTagRecycle"]
                tag = unit_tag(tagIndex, tagRecycle)
                name = event["m_unitTypeName"]
                gameloop = event["_gameloop"]
                player = event["m_controlPlayerId"]
                x = event["m_x"]
                y = event["m_y"]
                startingPosition = [UnitPosition(gameloop, x, y)]
                lifetime = UnitLifetime(tag, player, name, startingPosition, False, -1)
                units[tagIndex] = lifetime
                pastIndexHolders[tagIndex].append(lifetime)
            elif event["_event"] == "NNet.Replay.Tracker.SUnitDiedEvent":
                tagIndex = event["m_unitTagIndex"]
                gameloop = event["_gameloop"]
                killerIndex = event["m_killerUnitTagIndex"]

                killerTag = None
                # Fetch from past holders if current holder is dead
                # Handles case where adept dies before phase shift ends
                # TODO: Handle case where a unit dies to a building, or
                # when MULE timer runs out and dies to command center
                if units.get(killerIndex) is None:
                    # Temporary behaviour, -2 if killer is unknown.
                    killerTag = (
                        pastIndexHolders[killerIndex][-1].tag
                        if killerIndex in pastIndexHolders
                        else -2
                    )
                else:
                    killerTag = units[killerIndex].tag

                x = event["m_x"]
                y = event["m_y"]
                lastPos = UnitPosition(gameloop, x, y)

                # Exclude building death events
                if units.get(tagIndex) is None:
                    continue

                deadUnit = units[tagIndex]

                del units[tagIndex]  # Free up index

                deadUnit.positions.append(lastPos)
                deadUnit.dead = True
                deadUnit.killer = killerTag

                deadUnits.append(deadUnit)
            elif event["_event"] == "NNet.Replay.Tracker.SUnitPositionsEvent":
                gameloop = event["_gameloop"]
                tagIndex = event["m_firstUnitIndex"]
                items = event["m_items"]
                for i in range(0, len(items), 3):
                    tagIndex += items[i]
                    x = items[i + 1]
                    y = items[i + 2]
                    pos = UnitPosition(gameloop, x, y)
                    # TODO: Planetary Fortress can be included in SUnitPositionsEvents due to their
                    # ability to do damage. Excluding them for now, need better way to handle this.
                    if tagIndex not in units:
                        continue
                    units[tagIndex].positions.append(pos)

        deadUnits.extend(list(units.values()))
        return deadUnits
