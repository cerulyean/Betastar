from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId







WORKERS = [UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE]
NOT_ARMY = WORKERS + [UnitTypeId.OVERLORD, UnitTypeId.OVERSEER, UnitTypeId.OVERLORDTRANSPORT, UnitTypeId.LARVA, UnitTypeId.EGG]

from sc2.ids.unit_typeid import UnitTypeId
from sc2.data import Race

VALID_UNITS = {
    Race.Zerg: {
        # Workers and Larva
        UnitTypeId.DRONE,
        UnitTypeId.LARVA,
        UnitTypeId.EGG,

        # Army units
        UnitTypeId.ZERGLING,
        UnitTypeId.BANELING,
        UnitTypeId.ROACH,
        UnitTypeId.RAVAGER,
        UnitTypeId.HYDRALISK,
        UnitTypeId.LURKERMP,
        UnitTypeId.INFESTOR,
        UnitTypeId.SWARMHOSTMP,
        UnitTypeId.ULTRALISK,
        UnitTypeId.VIPER,

        # Air units
        UnitTypeId.MUTALISK,
        UnitTypeId.CORRUPTOR,
        UnitTypeId.BROODLORD,
        UnitTypeId.OVERSEER,

        # Support units
        UnitTypeId.QUEEN,
        UnitTypeId.OVERLORD,

        # Hatchery morphs
        UnitTypeId.HATCHERY,
        UnitTypeId.LAIR,
        UnitTypeId.HIVE,

        # Basic economy
        UnitTypeId.EXTRACTOR,

        # Base defenses
        UnitTypeId.SPINECRAWLER,
        UnitTypeId.SPORECRAWLER,

        # Core structures
        UnitTypeId.SPAWNINGPOOL,
        UnitTypeId.EVOLUTIONCHAMBER,

        # Tech structures
        UnitTypeId.BANELINGNEST,
        UnitTypeId.ROACHWARREN,
        UnitTypeId.HYDRALISKDEN,
        UnitTypeId.LURKERDENMP,
        UnitTypeId.INFESTATIONPIT,
        UnitTypeId.SPIRE,
        UnitTypeId.GREATERSPIRE,
        UnitTypeId.ULTRALISKCAVERN,

        # Macro & transport
        UnitTypeId.NYDUSNETWORK,
        UnitTypeId.NYDUSCANAL,
    }
}
