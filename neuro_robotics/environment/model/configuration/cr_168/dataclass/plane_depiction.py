from attrs import define
from attrs import validators


@define
class PlaneDepiction:

    description_file_location: str
    init_position: list
