"""pytest plugin: a Python socket guard that refuses non-loopback lookups and
connections for the whole session.

Used by the "player-props-py312" CI job (``-p tests.no_network_plugin``) so a
test that reaches the network through Python's ``socket`` module fails instead
of downloading anything.

Allowed: the hostname ``localhost``, literal loopback addresses (127.0.0.0/8,
``::1``, IPv4-mapped ``::ffff:127.x.x.x``), the unspecified addresses
(``0.0.0.0``, ``::``, ``''``), ``None`` and non-IP sockets (Unix sockets).
Every other hostname is refused before any DNS resolution.

Patched: ``socket.getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``,
``gethostbyaddr``, ``create_connection``, ``socket.socket.connect`` and
``connect_ex``; all are restored at ``pytest_unconfigure``.

This is not OS-level network isolation. Code that bound one of these functions
before the plugin loaded (``from socket import getaddrinfo``), C extensions with
their own networking, and subprocesses are not covered.
"""
import ipaddress
import socket

UNSPECIFIED = {'', '0.0.0.0', '::'}
PATCHED = ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'create_connection')
PATCHED_METHODS = ('connect', 'connect_ex')


class NetworkBlocked(RuntimeError):
    pass


def is_local(host):
    """True for None, 'localhost', unspecified and literal loopback addresses."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode('ascii', 'replace')
    if not isinstance(host, str):
        return False
    if host.lower() == 'localhost' or host in UNSPECIFIED:
        return True
    literal = host[1:-1] if host.startswith('[') and host.endswith(']') else host
    try:
        addr = ipaddress.ip_address(literal.split('%', 1)[0])     # drop an IPv6 zone id
    except ValueError:
        return False                                              # any other hostname
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped        # some 3.12 releases report ::ffff:127.0.0.1 as not loopback
    return addr.is_loopback or addr.is_unspecified


def _check(host, what):
    if not is_local(host):
        raise NetworkBlocked(f"network access blocked during tests: {what} {host!r}")


def _address_host(address):
    if isinstance(address, tuple) and address:
        return address[0]
    return address                   # e.g. a bare string: refused unless local


_saved = {}


def pytest_configure(config):
    if _saved:
        return
    _saved.update({name: getattr(socket, name) for name in PATCHED})
    _saved.update({'socket.' + name: getattr(socket.socket, name) for name in PATCHED_METHODS})

    def getaddrinfo(host, *a, **k):
        _check(host, 'look up')
        return _saved['getaddrinfo'](host, *a, **k)

    def gethostbyname(host):
        _check(host, 'look up')
        return _saved['gethostbyname'](host)

    def gethostbyname_ex(host):
        _check(host, 'look up')
        return _saved['gethostbyname_ex'](host)

    def gethostbyaddr(host):
        _check(host, 'look up')
        return _saved['gethostbyaddr'](host)

    def create_connection(address, *a, **k):
        _check(_address_host(address), 'connect to')
        return _saved['create_connection'](address, *a, **k)

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(_address_host(address), 'connect to')
        return _saved['socket.connect'](self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(_address_host(address), 'connect to')
        return _saved['socket.connect_ex'](self, address)

    for fn in (getaddrinfo, gethostbyname, gethostbyname_ex, gethostbyaddr, create_connection):
        setattr(socket, fn.__name__, fn)
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


def pytest_unconfigure(config):
    for name, original in _saved.items():
        if name.startswith('socket.'):
            setattr(socket.socket, name[len('socket.'):], original)
        else:
            setattr(socket, name, original)
    _saved.clear()
