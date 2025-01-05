import inspect
from typing import Generic, Type, TypeVar

from gateforge.core import NetProxy, Reg, Wire

TBus = TypeVar("TBus", bound="Bus")
TInterface = TypeVar("TInterface", bound="Interface")


class Bus(Generic[TBus]):

    @classmethod
    def Create(cls: Type[TBus], **kwargs) -> TBus:
        annotations = inspect.get_annotations(cls)
        bus = cls()
        for name, netCls in annotations.items():
            if netCls is not Wire and netCls is not Reg:
                raise Exception(f"Bad type annotation for net `{name}`: `{netCls.__name__}`, "
                                "should be `Wire` or `Reg`")
            if name not in kwargs:
                raise Exception(f"Bus net `{name}` has not been specified")
            value = kwargs[name]
            if not isinstance(value, NetProxy):
                raise Exception(
                    f"`NetProxy` instance expected for net `{name}`, has `{type(value).__name__}")
            if netCls is not Reg and value.isReg:
                raise Exception(f"Expected `Wire` for net `{name}`, has `Reg`")
            if netCls is Reg and not value.isReg:
                raise Exception(f"Expected `Reg` for net `{name}`, has `Wire`")
            setattr(bus, name, value)
            if value.initialName is None:
                value.SetName(f"{name}")

        for name in kwargs.keys():
            if name not in annotations:
                raise Exception(f"Specified net `{name}` not present in tne bus class annotations")

        return bus


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
        return super(Interface, type(self)).Create(**values)


    def Assign(self, **kwargs):
        """
        Perform assignment on the specified bus nets. Assignment direction depends on bus nets
        declared direction.
        """
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise Exception(f"No bus net `{name}`")
            net = getattr(self, name)
            if net.isOutput:
                net <<= value
            else:
                value <<= net


class Interface(Bus[TInterface]):
    internal: TInterface
    external: TInterface


    @classmethod
    def Create(cls: Type[TInterface], **kwargs) -> TInterface:
        iface = super(Interface, cls).Create(**kwargs)
        iface.internal = iface
        iface.external = iface.Adjacent()
        return iface
