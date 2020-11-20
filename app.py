#!/usr/bin/env python3
# thoth-graph-estimator
# Copyright(C) 2020 Fridolin Pokorny
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# type: ignore

"""Thoth's graph estimator for checking dependency graph size."""

import logging
import os
import sys
from collections import deque
from typing import Optional
from typing import Set

import attr
import click
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import sessionmaker
from thoth.storages import __version__ as thoth_storages_version
from thoth.common import init_logging
from thoth.common import __version__ as thoth_common_version
from thoth.python import Pipfile
from thoth.storages import GraphDatabase

__version__ = "0.0.0"
__component_version__ = f"{__version__}+storages.{thoth_storages_version}.common.{thoth_common_version}"

# logging.getLogger("sqlalchemy.engine.base.Engine").setLevel(logging.WARNING)
init_logging()

_BUCKET_LARGE_SIZE = 3
_BUCKET_MEDIUM_SIZE = 2
_BUCKET_SMALL_SIZE = 1
_DISCOUNT_FACTOR = 100
_LOGGER = logging.getLogger("thoth.graph_estimator")
_RESOURCE_HUNGRY_RECOMMENDATION_TYPES = frozenset({"stable", "performance", "security", "testing"})

_ModelBase = declarative_base()


class Package(_ModelBase):
    """A table mapping package to graph size calculated."""

    __tablename__ = 'package'

    id = Column(Integer, primary_key=True)
    package_name = Column(String)
    version_count = Column(Integer)
    subgraph_size = Column(Float)

    def __repr__(self):
        """Representation of self."""
        return f'Package {self.package_name}'


@attr.s(slots=True)
class SubGraphEntity:
    """A class representing a subgraph that needs to be checked."""

    subgraph_name = attr.ib(type=str, init=True)

    subgraphs_seen = attr.ib(type=Set[str], factory=set)
    to_visit = attr.ib(type=Set[str], factory=set)
    subgraph_size = attr.ib(type=float, default=1.0)


def _print_version(ctx: click.Context, _, value: str):
    """Print graph estimator version and exit."""
    if not value or ctx.resilient_parsing:
        return

    click.echo(__version__)
    ctx.exit()


def _get_session(database_path: str) -> Session:
    """Create a database."""
    engine = create_engine(f"sqlite:///{database_path}", echo=True)
    _ModelBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _fill_version_count(graph: GraphDatabase, session: Session) -> None:
    """Compute number of versions stored in the database for each package."""
    _LOGGER.info("Checking number of versions for each package")

    for package_name in graph.get_python_package_version_names_all(distinct=True):
        version_count = graph.get_package_versions_count_all(package_name)
        entry = session.query(Package).filter(Package.package_name == package_name).first()
        if not entry:
            entry = Package(package_name=package_name)

        entry.version_count = version_count
        session.add(entry)
        session.commit()


def _fill_graph_score(graph: GraphDatabase, session: Session) -> None:
    """Compute and fill in graph score per each package."""
    _LOGGER.info("Computing graph score for each package")

    subgraphs = deque()

    # The very first walk will mark down libraries that do not have any dependencies.
    for package_name in graph.get_python_package_version_names_all(distinct=True):
        dependencies = graph.get_depends_on_package_names(package_name)
        subgraphs.append(SubGraphEntity(subgraph_name=package_name, to_visit=set(dependencies)))
        if not dependencies:
            entry = session.query(Package).filter(Package.package_name == package_name).first()
            if not entry:
                # Might be ingesting in the mean time, do not mark down and continue.
                continue

            entry.subgraph_size = entry.version_count
            session.commit()
        else:
            subgraphs.append(SubGraphEntity(subgraph_name=package_name, to_visit=set(dependencies)))

    while subgraphs:
        subgraph = subgraphs.popleft()

        for package_name in subgraph.to_visit:
            entry = session.query(Package).filter(Package.package_name == package_name).first()
            if not entry:
                _LOGGER.warning("Cannot score subgraph %r as not all the dependencies were resolved", package_name)
                break

            if entry.subgraph_size is None:
                # Scheduling for the next round.
                subgraphs.append(subgraph)
                break

            subgraph.subgraph_size *= entry.subgraph_size * entry.version_count
            subgraph.subgraphs_seen.add(package_name)
        else:
            entry = session.query(Package).filter(Package.package_name == subgraph.subgraph_name).first()
            if not entry:
                _LOGGER.error("No subgraph for %r found, this looks like a programming error")
                continue

            entry.subgraph_size = subgraph.subgraph_size
            session.commit()

        subgraph.to_visit -= subgraph.subgraphs_seen


@click.group()
@click.pass_context
@click.option(
    "-v", "--verbose", is_flag=True, envvar="THOTH_GRAPH_ESTIMATOR_DEBUG", help="Be verbose about what's going on.",
)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    callback=_print_version,
    expose_value=False,
    help="Print graph estimator version and exit.",
)
def cli(ctx: Optional[click.Context] = None, verbose: bool = False):
    """Thoth' graph estimator command line interface."""

    if ctx:
        ctx.auto_envvar_prefix = "THOTH_GRAPH_ESTIMATOR"

    if verbose:
        _LOGGER.setLevel(logging.DEBUG)

    _LOGGER.debug("Debug mode is on")


@cli.command()
@click.option(
    "--path",
    type=str,
    envvar="THOTH_GRAPH_ESTIMATOR_PATH",
    default="data.db",
    metavar="FILE.db",
    help="Path to graph estimator output data file.",
)
@click.option(
    "--recreate",
    type=str,
    envvar="THOTH_GRAPH_ESTIMATOR_RECREATE",
    is_flag=True,
    default=False,
    help="Recreate the database - remove old one and create a new fresh database.",
)
def pile(path: str, recreate: bool) -> None:
    """Pile a local cache for the dependency graph size estimation."""
    _LOGGER.info("Connecting to the database...")
    graph = GraphDatabase()
    graph.connect()
    _LOGGER.info("Connection to the database done")

    if os.path.isfile(path) and recreate:
        _LOGGER.warning("Removing old database file %r", path)
        os.remove(path)

    session = _get_session(path)

    # _fill_version_count(graph, session)
    _fill_graph_score(graph, session)


def _do_estimate(recommendation_type: str, pipfile: Pipfile) -> None:
    """Estimate size of the bucket based on inputs provided."""


@cli.command()
@click.option(
    "--recommendation-type",
    type=str,
    envvar="THOTH_GRAPH_ESTIMATOR_RECOMMENDATION_TYPE",
    default="latest",
    show_default=True,
    help="Type of recommendation generated based on knowledge base.",
)
@click.option(
    "--requirements",
    type=str,
    envvar="THOTH_GRAPH_ESTIMATOR_REQUIREMENTS",
    required=True,
    metavar="PIPFILE",
    show_default=True,
    help="Type of recommendation generated based on knowledge base.",
)
def estimate(recommendation_type: str, pipfile: str) -> None:
    """Estimate how big the dependency graph would be."""
    if recommendation_type == "latest":
        sys.exit(_BUCKET_SMALL_SIZE)
    elif recommendation_type in _RESOURCE_HUNGRY_RECOMMENDATION_TYPES:
        if os.path.isfile(pipfile):
            pipfile_instance = Pipfile.from_file(pipfile)
        else:
            pipfile_instance = Pipfile.from_string(pipfile)

        _do_estimate(recommendation_type, pipfile_instance)
    else:
        _LOGGER.error("Unknown recommendation type %r, assuming largest bucket size", recommendation_type)
        sys.exit(_BUCKET_LARGE_SIZE)


__name__ == "__main__" and cli()
