import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self._engine_users = 0
        self._registered_ptrs: list[int] = []

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        with self.transfer_engine_lock:
            if self.transfer_engine is None:
                try:
                    from mooncake.engine import TransferEngine  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Please install mooncake by following the instructions at "
                        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                        "to run vLLM with MooncakeConnector."
                    ) from e
                self.transfer_engine = TransferEngine()
                device_name = device_name if device_name is not None else ""
                ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                if ret_value != 0:
                    raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
            self._engine_users += 1
            return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
                self._registered_ptrs.append(ptr)
            self.is_register_buffer = True

    def release_transfer_engine(self, engine) -> None:
        """Keep shared engines live until their last explicit owner shuts down."""
        with self.transfer_engine_lock:
            if engine is not self.transfer_engine or self._engine_users == 0:
                raise RuntimeError("Mooncake transfer engine is not owned by this registry")
            if self._engine_users > 1:
                self._engine_users -= 1
                return
            with self.register_buffer_lock:
                # Retain unprocessed registrations if the SDK reports a failure.
                while self._registered_ptrs:
                    ptr = self._registered_ptrs[-1]
                    ret_value = engine.unregister_memory(ptr)
                    if ret_value != 0:
                        raise RuntimeError(f"Mooncake memory unregistration failed: {ret_value}")
                    self._registered_ptrs.pop()
                self.is_register_buffer = False
                self.transfer_engine = None
                self._engine_users = 0


global_te = GlobalTE()
