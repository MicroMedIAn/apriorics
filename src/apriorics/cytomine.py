from os import PathLike
from pathlib import Path
from typing import List, Optional

import shapely
from shapely.geometry import MultiPolygon

from cytomine import Cytomine
from cytomine.models import (
    Annotation,
    AnnotationCollection,
    AnnotationTerm,
    ImageInstanceCollection,
    Project,
    StorageCollection,
    TermCollection,
)
from cytomine.models.collection import CollectionPartialUploadException


def get_image_id(slidename: str, id_project: int) -> int:
    r"""
    Fetch cytomine image id corresponding to a slide filename. Raises a RuntimeError if
    image is not found.

    Args:
        slidename: annotated slide's filename.
        id_project: cytomine project's id.

    Returns:
        Image id
    """
    images = ImageInstanceCollection().fetch_with_filter("project", id_project)
    images = filter(lambda x: slidename in x.filename, images)

    try:
        id_image = next(images).id
    except StopIteration:
        print(
            f"Slide {slidename} was not found in cytomine, please upload it. "
            "Stopping program..."
        )
        raise RuntimeError

    return id_image


def get_term_id(term: str, id_project: int) -> int:
    r"""
    Fetch cytomine term id corresponding to a given term.

    Args:
        term: input term.
        id_project: cytomine project's id.

    Returns:
        Term id, or `None` if not found.
    """
    if term is not None:
        terms = TermCollection().fetch_with_filter("project", id_project)
        terms = filter(lambda x: x.name == term, terms)

        try:
            id_term = next(terms).id
        except StopIteration:
            print(
                f"Term {term} was not found on cytomine, please upload it. "
                "Resuming with no term..."
            )
            id_term = None
    else:
        id_term = None

    return id_term


def polygons_to_annotations(
    polygons: MultiPolygon,
    id_image: int,
    id_project: int,
    object_min_size: int = 1000,
    polygon_type: str = "polygon",
) -> AnnotationCollection:
    r"""
    Converts a shapely `MultiPolygon` object into a cytomine ready-to-upload
    `AnnotationCollection` object. Can optionally filter small polygons or replace them
    with their bounding box.

    Args:
        polygons: `MultiPolygon` object to upload.
        id_image: cytomine id of the image to annotate.
        id_project: cytomine project id.
        object_min_size: minimum accepted area of a polygon.
        polygon_type: if "polygon", upload polygons without changes; if "box", upload
            their bounding boxes instead.

    Returns:
        The collection of cytomine annotations to upload.
    """
    annotations = AnnotationCollection()
    for polygon in polygons.geoms:
        if polygon.area < object_min_size:
            continue
        if polygon_type == "box":
            bbox = shapely.geometry.box(*polygon.bounds)
            location = bbox.wkt
        else:
            location = polygon.wkt
        annotations.append(
            Annotation(
                location=location,
                id_image=id_image,
                id_project=id_project,
            )
        )
    return annotations


def upload_annotations_with_terms(
    annotations: AnnotationCollection,
    id_image: int,
    id_project: int,
    id_term: Optional[int] = None,
):
    r"""
    Upload annotations to cytomine, optionally with given term.

    Args:
        annotations: collection of annotations to upload.
        id_image: cytomine id of the image to annotate.
        id_project: cytomine project id.
        id_term: cytomine id of the term to use for annotation.
    """
    try:
        annotations.save()
    except CollectionPartialUploadException as e:
        print(e)
    if id_term is not None:
        annotations = AnnotationCollection()
        annotations.project = id_project
        annotations.image = id_image
        annotations.fetch()
        for annotation in annotations:
            if not annotation.term:
                AnnotationTerm(annotation.id, id_term).save()


def upload_polygons_to_cytomine(
    polygons: MultiPolygon,
    slidename: str,
    host: str,
    public_key: str,
    private_key: str,
    id_project: int,
    term: Optional[str] = None,
    polygon_type: str = "polygon",
    object_min_size: int = 1000,
):
    r"""
    Upload polygons to cytomine. Can optionally specify a term, filter small polygons or
    upload bounding boxes instead.

    Args:
        polygons: `MultiPolygon` object to upload.
        slidename: annotated slide's filename.
        host: cytomine core ip address.
        public_key: cytomine API public key.
        private_key: cytomine API private key.
        id_project: cytomine project id.
        term: term to use for annotation.
        polygon_type: if "polygon", upload polygons without changes; if "box", upload
            their bounding boxes instead.
        object_min_size: minimum accepted area of a polygon.
    """
    with Cytomine(host=host, public_key=public_key, private_key=private_key) as _:
        id_image = get_image_id(slidename, id_project)
        id_term = get_term_id(term, id_project)

        annotations = polygons_to_annotations(
            polygons,
            id_image,
            id_project,
            object_min_size=object_min_size,
            polygon_type=polygon_type,
        )

        upload_annotations_with_terms(
            annotations, id_image, id_project, id_term=id_term
        )


def upload_image_to_cytomine(
    filepath: PathLike, host: str, public_key: str, private_key: str, id_project: str
):
    r"""
    Upload an image to a given cytomine project.

    Args:
        filepath: path to image file.
        host: cytomine core ip address.
        public_key: cytomine API public key.
        private_key: cytomine API private key.
        id_project: cytomine project id.
    """
    filepath = Path(filepath)

    with Cytomine(
        host=host,
        public_key=public_key,
        private_key=private_key,
    ) as cytomine:

        # Check that the file exists on your file system
        if not filepath.exists():
            raise ValueError("The file you want to upload does not exist")

        # Check that the given project exists
        if id_project:
            project = Project().fetch(id_project)
            if not project:
                raise ValueError("Project not found")

        # To upload the image, we need to know the ID of your Cytomine storage.
        storages = StorageCollection().fetch()
        my_storage = next(
            filter(lambda storage: storage.user == cytomine.current_user.id, storages)
        )
        if not my_storage:
            raise ValueError("Storage not found")

        uploaded_file = cytomine.upload_image(
            upload_host="http://localhost-upload",
            filename=str(filepath),
            id_storage=my_storage.id,
            id_project=id_project,
        )

        print(uploaded_file)


def get_uploaded_images(
    host: str, public_key: str, private_key: str, id_project: int
) -> List[str]:
    r"""
    Get a list of uploaded images on a given cytomine project.

    Args:
        host: cytomine core ip address.
        public_key: cytomine API public key.
        private_key: cytomine API private key.
        id_project: cytomine project id.

    Returns:
        All uploaded images' filenames.
    """
    with Cytomine(host=host, public_key=public_key, private_key=private_key) as _:
        images = ImageInstanceCollection().fetch_with_filter("project", id_project)
    return [image.instanceFilename for image in images]
