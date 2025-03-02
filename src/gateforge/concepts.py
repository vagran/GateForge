import inspect
from typing import Any, Generic, Type, TypeVar

from gateforge.core import Dimensions, InputOutputTypeTag, Net, Port, RawExpression, NetProxy, \
    ParseException, Reg, TypeTag, Wire

TBus = TypeVar("TBus", bound="Bus")
TInterface = TypeVar("TInterface", bound="Interface")


class Bus(Generic[TBus]):

    @classmethod
    def Create(cls: Type[TBus], **kwargs: NetProxy) -> TBus:
        """Factory function for object instantiation. Expects all the member nets as keyword
        arguments. Note that `__init__()` method is not called. Use `Construct()` method if
        implementing constructor.

        :param kwargs: Fields of the bus instance.
        :return:
        """
        return cls._Create(**kwargs) # type: ignore


    @classmethod
    def CreateDefault(cls: Type[TBus], **kwargs: NetProxy) -> TBus:
        """Factory function for object instantiation. May override some member nets by keyword
        arguments. Note that `__init__()` method is not called. Use `ConstructDefault()` method if
        implementing constructor.

        :param kwargs: Fields of the bus instance.
        :return:
        """
        return cls._Create(_isDefault=True, **kwargs) # type: ignore


    def Adjacent(self):
        annotations = inspect.get_annotations(type(self))
        values = dict()
        for name, netCls in annotations.items():
            if not Bus._IsNetMember(netCls):
                continue
            value = getattr(self, name)
            if value.isOutput:
                adj = value.src.input
            else:
                adj = value.src.output
            values[name] = adj
        return super(Interface, type(self))._Create(_isAdjacent=True, **values)


    def Assign(self, **kwargs: RawExpression):
        """
        Perform assignment on the specified bus nets. Assignment direction depends on bus nets
        declared direction.
        """
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise ParseException(f"No bus net `{name}`")
            net = getattr(self, name)
            if net.isOutput:
                net.assign(value, frameDepth=1)
            else:
                value.assign(net, frameDepth=1) # type: ignore


    def Construct(self, **kwargs: NetProxy):
        return self._Construct(**kwargs) # type: ignore


    def ConstructDefault(self, **kwargs: NetProxy):
        return self._Construct(_isDefault=True, **kwargs) # type: ignore


    @classmethod
    def _Create(cls: Type[TBus], *, _isAdjacent: bool = False,  _isDefault: bool = False,
                _frameDepth: int = 1, **kwargs: NetProxy) -> TBus:
        bus = cls.__new__(cls)
        bus._Construct(_isAdjacent=_isAdjacent, _isDefault=_isDefault, _frameDepth=_frameDepth + 1,
                       **kwargs)
        return bus


    def _Construct(self, *, _isAdjacent: bool = False, _isDefault: bool = False,
                   _frameDepth: int = 1, **kwargs: NetProxy):
        if _isAdjacent and _isDefault:
            raise Exception("Adjacent and default flags cannot be set simultaneously")
        annotations = inspect.get_annotations(type(self))
        for name, tag in annotations.items():
            if not Bus._IsNetMember(tag):
                continue
            value: Net
            if name not in kwargs:
                if _isDefault:
                    net = tag.cls(dims=tag.dims,
                                  isReg=tag.cls is Reg, name=name, frameDepth=_frameDepth + 1)
                    value = net.output if tag.isOutput else net.input
                    setattr(self, name, value)
                    continue
                raise ParseException(f"Bus net `{name}` has not been specified")
            value = kwargs[name]
            if isinstance(value, Port):
                value = value.output if value.isOutput else value.input
            if not isinstance(value, NetProxy):
                raise ParseException(
                    f"`NetProxy` or `Port` instance expected for net `{name}`, has `{type(value).__name__}")
            if tag.cls is not Reg and value.isReg:
                raise ParseException(f"Expected `Wire` for net `{name}`, has `Reg`")
            if tag.cls is Reg and not value.isReg:
                raise ParseException(f"Expected `Reg` for net `{name}`, has `Wire`")
            if not _isAdjacent and tag.isOutput != value.isOutput:
                raise ParseException(f"Net `{name}` direction mismatch")
            if _isAdjacent and tag.isOutput == value.isOutput:
                raise Exception(f"Unexpected adjacent direction for net `{name}`")
            if tag.dims is not None and not Dimensions.MatchAny(tag.dims, value.dims):
                raise ParseException(
                    f"Unexpected shape for net `{name}`: "
                    f"{Dimensions.StrAny(value.dims)} != {Dimensions.StrAny(tag.dims)}")
            setattr(self, name, value)
            if value.initialName is None:
                value.SetName(f"{name}")

        for name in kwargs.keys():
            if name not in annotations:
                raise ParseException(f"Specified net `{name}` not present in tne bus class annotations")


    @staticmethod
    def _IsNetMember(annotation):
        return isinstance(annotation, InputOutputTypeTag)


class Interface(Bus[TInterface]):
    internal: TInterface
    external: TInterface


    @classmethod
    def Create(cls: Type[TInterface], **kwargs: NetProxy) -> TInterface:
        iface = super(Interface, cls)._Create(**kwargs) # type: ignore
        iface.internal = iface
        iface.external = iface.Adjacent()
        return iface


    @classmethod
    def CreateDefault(cls: Type[TInterface], **kwargs: NetProxy) -> TInterface:
        iface = super(Interface, cls)._Create(_isDefault=True, **kwargs) # type: ignore
        iface.internal = iface
        iface.external = iface.Adjacent()
        return iface


    def Construct(self, **kwargs: NetProxy):
        super()._Construct(**kwargs) # type: ignore
        self.internal = self # type: ignore
        self.external = self.Adjacent()


    def ConstructDefault(self, **kwargs: NetProxy):
        super()._Construct(_isDefault=True, **kwargs) # type: ignore
        self.internal = self # type: ignore
        self.external = self.Adjacent()


def ConstructNets(obj: Any):
    """Construct all non-initialized annotated Net class members in the specified object. Field name
    is used as net name. Active namespaces affect names as usual.

    :param obj: Object which has some members annotated with `Reg` or `Wire` types. They are
    constructed if the object does not currently have such attribute. Size specification is taken
    into account as well.
    """

    annotations = inspect.get_annotations(type(obj))
    allowedTypes = [Wire, Reg]
    for name, netCls in annotations.items():
        tag = TypeTag.CheckAnnotation(netCls, allowedTypes)
        if tag is None:
            continue
        if hasattr(obj, name):
            continue
        net = tag.cls(dims=tag.dims, isReg=tag.cls is Reg, name=name, frameDepth=1)
        setattr(obj, name, net)
