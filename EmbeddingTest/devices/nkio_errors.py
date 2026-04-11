from __future__ import annotations


NKIO_ENOERR = 0
NKIO_EBUSY = 1
NKIO_EINUSED = 2
NKIO_EIO = 3
NKIO_ETIMEOUT = 4
NKIO_EDEVERR = 5
NKIO_EINVAL = 6
NKIO_EFILE = 7


class NkioError(RuntimeError):
    """Base exception for NK I/O board failures."""

    def __init__(self, code: int, operation: str, detail: str | None = None) -> None:
        self.code = int(code)
        self.operation = operation
        self.detail = detail or ""
        message = f"{operation} failed with NKIO error {self.code}: {nkio_error_name(self.code)}"
        if self.detail:
            message += f" ({self.detail})"
        super().__init__(message)


class NkioBusyError(NkioError):
    pass


class NkioInUseError(NkioError):
    pass


class NkioIoError(NkioError):
    pass


class NkioTimeoutError(NkioError):
    pass


class NkioDeviceError(NkioError):
    pass


class NkioInvalidArgError(NkioError):
    pass


class NkioFileError(NkioError):
    pass


_ERROR_CLASS_BY_CODE = {
    NKIO_EBUSY: NkioBusyError,
    NKIO_EINUSED: NkioInUseError,
    NKIO_EIO: NkioIoError,
    NKIO_ETIMEOUT: NkioTimeoutError,
    NKIO_EDEVERR: NkioDeviceError,
    NKIO_EINVAL: NkioInvalidArgError,
    NKIO_EFILE: NkioFileError,
}

_ERROR_NAME_BY_CODE = {
    NKIO_ENOERR: "NKIO_ENOERR",
    NKIO_EBUSY: "NKIO_EBUSY",
    NKIO_EINUSED: "NKIO_EINUSED",
    NKIO_EIO: "NKIO_EIO",
    NKIO_ETIMEOUT: "NKIO_ETIMEOUT",
    NKIO_EDEVERR: "NKIO_EDEVERR",
    NKIO_EINVAL: "NKIO_EINVAL",
    NKIO_EFILE: "NKIO_EFILE",
}


def nkio_error_name(code: int) -> str:
    return _ERROR_NAME_BY_CODE.get(int(code), f"NKIO_UNKNOWN_{int(code)}")


def raise_for_code(code: int, operation: str, detail: str | None = None) -> None:
    code = int(code)
    if code == NKIO_ENOERR:
        return
    exc_type = _ERROR_CLASS_BY_CODE.get(code, NkioError)
    raise exc_type(code=code, operation=operation, detail=detail)
