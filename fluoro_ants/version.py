"""Release identity, kept separate from the algorithm version recorded in recipes."""

FLUOROANTS_RELEASE_VERSION = "2.12.0"

AUTHOR = "hx_mao"
# The address is put together where it is shown, so that harvesters scanning
# published source files for addresses do not find it in one piece.
_CONTACT = ("hengxu2010", "qq.com")


def contact_address() -> str:
    return "@".join(_CONTACT)
