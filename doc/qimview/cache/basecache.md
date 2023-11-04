Module qimview.cache.basecache
==============================

Classes
-------

`BaseCache(name: str = '')`
:   Abstract base class for generic types.
    
    A generic type is typically declared by inheriting from
    this class parameterized with one or more type variables.
    For example, a generic mapping type might be defined as::
    
      class Mapping(Generic[KT, VT]):
          def __getitem__(self, key: KT) -> VT:
              ...
          # Etc.
    
    This class can then be used as follows::
    
      def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
          try:
              return mapping[key]
          except KeyError:
              return default

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * qimview.cache.filecache.FileCache
    * qimview.cache.imagecache.ImageCache

    ### Methods

    `append(self, id: ~TId, value: ~TValue, extra: ~TExtra, check_size=True) ‑> None`
    :   :param id: cache element identifier
        :param value: cache value, typically numpy array of the image
        :param extra: additional data in the cache
        :return:

    `check_size_limit(self, update_progress: bool = False) ‑> None`
    :

    `get_cache_size(self) ‑> int`
    :

    `remove(self, id: ~TId) ‑> bool`
    :   Remove id from cache
        returns: True if removed False otherwise (not found)

    `reset(self) ‑> None`
    :

    `search(self, id: ~TId) ‑> Optional[Tuple[~TId, ~TValue, ~TExtra]]`
    :

    `set_max_cache_size(self, size: int) ‑> None`
    :

    `set_memory_bar(self, progress_bar: PySide6.QtWidgets.QProgressBar) ‑> None`
    :

    `update_progress(self)`
    :