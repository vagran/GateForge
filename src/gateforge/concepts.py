import inspect
from typing import Generic, Type, TypeVar

from gateforge.core import NetMarkerType, NetProxy, ParseException, Reg, Wire

TBus = TypeVar("TBus", bound="Bus")
TInterface = TypeVar("TInterface", bound="Interface")


class Bus(Generic[TBus]):

    @classmethod
    def Create(cls: Type[TBus], **kwargs) -> TBus:
        return cls._Create(False, **kwargs)


    def Adjacent(self):
        annotations = inspect.get_annotations(type(self))
        values = dict()
        for name in annotations.keys():
            value = getattr(self, name)
            if value.isOutput:
                adj = value.src.input
            else:
                adj = value.src.output
            values[name] = adj
        return super(Interface, type(self))._Create(True, **values)


    def Assign(self, **kwargs):
        """
        Perform assignment on the specified bus nets. Assignment direction depends on bus nets
        declared direction.
        """
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise ParseException(f"No bus net `{name}`")
            net = getattr(self, name)
            if net.isOutput:
                net <<= value
            else:
                value <<= net


    @classmethod
    def _Create(cls: Type[TBus], __isAdjacent: bool, **kwargs) -> TBus:
        annotations = inspect.get_annotations(cls)
        bus = cls()
        for name, netCls in annotations.items():
            if not isinstance(netCls, NetMarkerType):
                raise ParseException(f"Bad type annotation for net `{name}`: `{netCls}`, "
                                     "should be `NetMarkerType[Reg|Wire]`")
            if name not in kwargs:
                raise ParseException(f"Bus net `{name}` has not been specified")
            value = kwargs[name]
            if not isinstance(value, NetProxy):
                raise ParseException(
                    f"`NetProxy` instance expected for net `{name}`, has `{type(value).__name__}")
            if netCls.netType is not Reg and value.isReg:
                raise ParseException(f"Expected `Wire` for net `{name}`, has `Reg`")
            if netCls.netType is Reg and not value.isReg:
                raise ParseException(f"Expected `Reg` for net `{name}`, has `Wire`")
            if not __isAdjacent and netCls.isOutput != value.isOutput:
                raise ParseException(f"Net `{name}` direction mismatch")
            if __isAdjacent and netCls.isOutput == value.isOutput:
                raise Exception(f"Unexpected adjacent direction for net `{name}`")
            setattr(bus, name, value)
            if value.initialName is None:
                value.SetName(f"{name}")

        for name in kwargs.keys():
            if name not in annotations:
                raise ParseException(f"Specified net `{name}` not present in tne bus class annotations")

        return bus


class Interface(Bus[TInterface]):
    internal: TInterface
    external: TInterface


    @classmethod
    def Create(cls: Type[TInterface], **kwargs) -> TInterface:
        iface = super(Interface, cls).Create(**kwargs)
        iface.internal = iface
        iface.external = iface.Adjacent()
        return iface
