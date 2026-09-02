"""Registration idioms are uniform across all three registries."""
import torch
import autolrp as A


def _dummy(node, config):
    return None


class TestRegistrationIdioms:
    def test_installer_call_form(self):
        A.register_installer('FakeOpBackward', _dummy)
        try:
            assert A.INSTALLERS['FakeOpBackward'] is _dummy
        finally:
            del A.INSTALLERS['FakeOpBackward']

    def test_installer_decorator_form(self):
        @A.register_installer('FakeOp2Backward')
        def h(node, config):
            return None
        try:
            assert A.INSTALLERS['FakeOp2Backward'] is h
        finally:
            del A.INSTALLERS['FakeOp2Backward']

    def test_analyzer_and_rewrite_decorators(self):
        @A.register_analyzer('fake_fact')
        def an(nodes):
            return {}

        @A.register_rewrite('fake_fn')
        def rw(func, args, kwargs):
            return NotImplemented
        try:
            assert A.ANALYZERS['fake_fact'] is an
            assert A.REWRITES['fake_fn'] is rw
        finally:
            del A.ANALYZERS['fake_fact']
            del A.REWRITES['fake_fn']
