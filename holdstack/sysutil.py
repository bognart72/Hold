"""Rendszerszintű segédfüggvények."""
import sys


def disable_power_throttling():
    """Windows 10/11: a folyamat kivonja magát az energiatakarékos visszafogás (Power Throttling,
    EcoQoS) alól. Az ablak nélküli háttérfolyamatokat (pl. a GUI-ból indított feldolgozást) a
    Windows egyébként lassabb órajelen/hatékonysági magokon futtathatja. Más rendszeren nem csinál semmit."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4
        st = PROCESS_POWER_THROTTLING_STATE(PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                            PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        # pontos típusok: 64 biten a HANDLE különben csonkolódik, és a hívás hibát ad
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        ok = k32.SetProcessInformation(k32.GetCurrentProcess(), ProcessPowerThrottling,
                                       ctypes.byref(st), ctypes.sizeof(st))
        return bool(ok)
    except Exception:
        return False
