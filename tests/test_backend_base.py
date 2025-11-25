from latex_parser.backend_base import BackendBase


def test_backend_base_is_abstract():
    class Dummy(BackendBase):
        def _make_cache(self, config, options): ...

        def _compile_static(self, ir, cache, params, options=None): ...

        def _compile_time_dependent(
            self, ir, cache, params, *, t_name, time_symbols, options=None
        ): ...

    # Instantiation should succeed once abstract methods are implemented
    Dummy()
