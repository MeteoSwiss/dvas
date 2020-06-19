"""
This file contains testing classes and function for
dvas.dvas_helper module.

"""

# Import from python packages and modules
from pathlib import Path
from datetime import datetime
from abc import ABC
import pytest
from pampy import ANY

# Import from current package
from dvas.dvas_helper import SingleInstanceMetaClass
from dvas.dvas_helper import RequiredAttrMetaClass
from dvas.dvas_helper import TypedProperty
from dvas.dvas_helper import get_by_path
from dvas.dvas_helper import check_path
from dvas.dvas_helper import check_datetime


def test_single_instance_metaclass():
    """Function used to test the metaclass SingleInstanceMetaClass

    The function tests:
        - Uniqueness of instances construct from this metaclass.

    """

    class SingleVar(metaclass=SingleInstanceMetaClass):
        """Singleton class implementation"""

    # Create many instances
    instances = [SingleVar() for _ in range(5)]

    # Get id
    ids = [id(arg) for arg in instances]

    # Test uniqueness
    assert len(set(ids)) == 1


def test_required_attr_metaclass():
    """Function used to test the metaclass RequiredAttrMetaClass

    The function tests:
        - Correct instantiated class with valid attributes;
        - Raise exception for bad attribute type;
        - Raise exception for bad attribute name.

    """

    class RequiredAttr(ABC, metaclass=RequiredAttrMetaClass):
        """Required attribute abstract class"""
        REQUIRED_ATTRIBUTES = {'my_attr': int}

    class OK(RequiredAttr):
        """OK required attribute implemented class"""
        def __init__(self):
            self.my_attr = 1

    class KO1(RequiredAttr):
        """KO required attribute implemented class"""
        def __init__(self):
            self.my_attr = 'a'

    class KO2(RequiredAttr):
        """KO required attribute implemented class"""
        def __init__(self):
            self.ko_attr = 1

    # Test OK
    assert isinstance(OK(), OK)

    # Test KO
    with pytest.raises(ValueError):
        KO1()

    with pytest.raises(ValueError):
        KO2()


def my_output_fct(x, length, start=0):
    """Return x[start:(start+len)]"""
    return x[start:(start + length)]


class TestTypedProperty:
    """Class used to test TypedProperty class"""

    class CheckClass:
        """Class used to do checks"""
        my_str = TypedProperty(str)
        my_upper_str = TypedProperty(str, lambda x: x.upper())
        my_list = TypedProperty([1, ANY])
        my_truncated_str = TypedProperty(
            str, my_output_fct, args=(2,), kwargs={'start': 3})
        my_matched_str = TypedProperty(
            TypedProperty.re_str_choice(['A', 'B']), lambda *x: x[0]
        )
        my_matched_str_ic = TypedProperty(
            TypedProperty.re_str_choice(['A', 'B'], ignore_case=True),
            lambda *x: x[0]
        )

    # Create instance
    inst = CheckClass()

    def test_type_check(self):
        """Method to test type checking"""
        self.inst.my_str = 'a'
        self.inst.my_list = [1, 2]
        self.inst.my_matched_str = 'A'
        self.inst.my_matched_str_ic = 'a'

        with pytest.raises(TypeError):
            self.inst.my_str = 1

        with pytest.raises(TypeError):
            self.inst.my_list = [2, 1]

        with pytest.raises(TypeError):
            self.inst.my_matched_str = 'C'

    def test_setter_fct(self):
        """Method used to test setter function"""
        self.inst.my_upper_str = 'a'
        assert self.inst.my_upper_str == 'A'

        self.inst.my_truncated_str = 'abcde'
        assert self.inst.my_truncated_str == 'de'

        self.inst.my_matched_str = 'A'
        assert self.inst.my_matched_str == 'A'


def test_get_by_path():
    """Function used to test the metaclass SingleInstanceMetaClass

    The function tests:
        - Get nested value/item in lists
        - Get nested value/item in dicts
        - Get nested value/item in mixed dicts/lists
        - Raise exception KeyError

    """

    # Lists
    assert get_by_path([1, [2, 3]], [1, 0]) == 2
    assert get_by_path([1, [2, 3]], [1]) == [2, 3]

    # Dicts
    assert get_by_path({'a': {'b': 1, 'c': 0}, 'b': 2}, ['a', 'c']) == 0
    assert (
        get_by_path(
            {'a': {'b': 1, 'c': 0}, 'b': 2}, ['a']
        ) == {'b': 1, 'c': 0}
    )

    # Dicts/Lists
    assert get_by_path({'a': [0, {'a': [10, 20]}]}, ['a', 1, 'a']) == [10, 20]

    # Raises
    for items in [[0, 10], [10]]:
        with pytest.raises((IndexError, TypeError)):
            get_by_path([1, [2, 3]], items)

    with pytest.raises(KeyError):
        get_by_path({'a': {'b': 1, 'c': 0}, 'b': 2}, ['a', 'z'])


@pytest.fixture(autouse=True)
def test_check_path(tmpdir):
    """Function to test check_path"""

    # Test str path name
    assert check_path(Path(tmpdir).as_posix()) == Path(tmpdir)

    # Test exist_ok True
    assert check_path(Path(tmpdir), exist_ok=True) == Path(tmpdir)

    # Test exist_ok False
    assert check_path(Path(tmpdir) / 'test') == Path(tmpdir) / 'test'

    # Raise exception
    with pytest.raises(TypeError):
        check_path(Path(tmpdir) / 'dummy', exist_ok=True)


def test_check_datetime():
    """Test dvas_helper.set_datetime"""

    assert check_datetime('20200101', utc=False) == datetime(2020, 1, 1)
    assert check_datetime(datetime(2020, 1, 1), utc=False) ==\
        datetime(2020, 1, 1)
    check_datetime('20200101T000000Z')

    with pytest.raises(TypeError):
        check_datetime('20200101')
