from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, Callable, Dict
from copy import deepcopy
import logging
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import dgl


from kedro.io.core import (
    PROTOCOL_DELIMITER,
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec


class DglDataset(AbstractVersionedDataSet, dgl.data.DGLDataset):  # ! Different from dgl.data import DGLDataset
    """``DglDataset`` loads and saves DGL graphs. It uses fsspec to handle
    the underlying filesystem operations.
    """

    DEFAULT_SAVE_ARGS = {"compress": "gzip"}  # type: Dict[str, Any]

    def __init__(
        self,
        filepath: str,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        """Creates a new instance of ``DglDataset`` pointing to a concrete
        DGL graph.
        Args:
            filepath: Path to a DGL graph.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded. If its ``save``
                attribute is None, save version will be autogenerated.
            credentials: Credentials required to get access to the underlying filesystem.
                E.g. for ``GCSFileSystem`` it should look like `{"token": None}`.
            fs_args: Extra arguments to pass into underlying filesystem class constructor
                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``), as well as
                to pass to the filesystem's `open` method through nested keys
                `open_args_load` and `open_args_save`. For example:
                `{"open_args_load": {"mode": "rb"}, "open_args_save": {"mode": "wb"}}`
            load_args: Extra arguments to pass to DGL's ``
            save_args: Extra arguments to pass to DGL's ``
        """

        _fs_args = deepcopy(fs_args) or {}
        _save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        _save_args.update(save_args or {})

        protocol, path = get_protocol_and_path(filepath, version)

        self._protocol = protocol
        self._fs = fsspec.filesystem(
            protocol, **_fs_args, **(credentials or {})
        )
        self._filepath = PurePosixPath(path)
        self._load_args = deepcopy(load_args) or {}
        self._save_args = _save_args
        if self._filepath:
            super().__init__(
                filepath=PurePosixPath(path),
                version=version,
                exists_function=self._fs.exists,
                glob_function=self._fs.glob,
            )

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def _load(self):
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        logging.info("Loading DGL graph from %s", load_path)
        with self._fs.open(load_path + '_dgl_graph.bin', **self._fs_open_args_load) as fs_file:
            graphs, labels_dict = load_graphs(fs_file, **self._load_args)
        with self._fs.open(load_path + '_info.pkl', **self._fs_open_args_load) as fs_file:
            info = load_info(fs_file)
        return (graphs, labels_dict['labels'], info['num_classes'])


    def _save(self, data) -> None:
        (graphs, labels, num_classes) = data
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        logging.info("Saving DGL graph to %s", save_path)
        save_graphs(save_path + '_dgl_graph.bin', graphs, {'labels': labels})
        save_info(save_path + '_info.pkl', {'num_classes': num_classes})

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        return os.path.exists(save_path + '_dgl_graph.bin') and os.path.exists(save_path + '_info.pkl')

    def _exists(self) -> bool:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)
