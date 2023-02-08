"""Paginator."""
from contextlib import contextmanager
from json import loads
from json.decoder import JSONDecodeError
from typing import Type, Generator, Any, get_type_hints, TypeVar, Callable
from fastapi.exceptions import HTTPException
from pydantic import parse_obj_as, parse_raw_as, ValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from databases import Database
from sqlalchemy import Column
from sqlalchemy.sql import Select, select, not_
from sqlalchemy.sql.functions import count
from fastapi_paginator._models import BaseModelT, Page, PageParameters


T = TypeVar("T")


class _UnprocessableEntity(HTTPException):
    """422 Unprocessable Entity."""

    def __init__(self, detail: Any, *, loc: tuple[str, ...] | None = None) -> None:
        if isinstance(detail, ValidationError):
            detail = detail.errors()
        elif isinstance(detail, Exception):
            detail = [dict(msg=str(detail))]
        else:
            detail = [dict(msg=detail)]
        if loc:
            detail[0]["loc"] = loc
        HTTPException.__init__(
            self,
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )


def _parse_args(parameter: str, field_type: Type[T]) -> tuple[T]:
    """Parse operator parameter and return operator method positional arguments.

    Case where a single scalar value is excepted.

    Args:
        parameter: Operator parameter to parse.
        field_type: Field data type.

    Returns:
        Arguments to pass to the operator method.
    """
    return (Paginator._parse_value(field_type, parameter),)


def _parse_args_list(parameter: str, field_type: Type[T]) -> tuple[list[T]]:
    """Parse operator parameter and return operator method positional arguments.

    Case where a list of values is excepted.

    Args:
        parameter: Operator parameter to parse.
        field_type: Field data type.

    Returns:
        Arguments to pass to the operator method.
    """
    return (
        Paginator._parse_value(
            list[field_type], f"[{parameter.strip('[]')}]"  # type: ignore
        ),
    )


def _parse_args_dual(parameter: str, field_type: Type[T]) -> list[T]:
    """Parse operator parameter and return operator method positional arguments.

    Case where a list of exactly 2 values is excepted.

    Args:
        parameter: Operator parameter to parse.
        field_type: Field data type.

    Returns:
        Arguments to pass to the operator method.
    """
    values = Paginator._parse_value(
        list[field_type],  # type: ignore
        f"[{parameter.strip('[]')}]",
    )
    if len(values) != 2:
        raise ValueError("Filter argument requires exactly 2 values.")
    return values


_OPERATOR_AUTO_ESCAPE = dict(autoescape=True)


def _get_default_exception() -> Type[BaseException]:
    """Get default exceptions to handle.

    Returns:
        Exception types.
    """
    return RuntimeError


def _get_asyncpg_exception() -> Type[BaseException]:
    """Get postgresql+asyncpg exceptions to handle.

    Returns:
        Exception types.
    """
    from asyncpg.exceptions import UndefinedFunctionError

    return UndefinedFunctionError


class Paginator:
    """Paginator."""

    __slots__ = ("_database", "_function_exceptions")

    #: JSON loads method, default to "json.loads". Override to use alternate library.
    json_loads: Callable[[bytes | bytearray | memoryview | str], Any] = loads

    #: Filters operators with SQLAlchemy equivalents
    _OPERATORS: dict[str, Any] = {
        "=": {"method": "__eq__"},
        "!=": {"method": "__ne__"},
        ">": {"method": "__gt__"},
        "<": {"method": "__lt__"},
        ">=": {"method": "__ge__"},
        "<=": {"method": "__le__"},
        "between": {"method": "between", "parser": _parse_args_dual},
        "in": {"method": "in_", "parser": _parse_args_list},
        "!in": {"method": "not_in", "parser": _parse_args_list},
        "like": {"method": "like", "value_type": str},
        "!like": {"method": "notlike", "value_type": str},
        "ilike": {"method": "ilike", "value_type": str},
        "!ilike": {"method": "notilike", "value_type": str},
        "startswith": {"method": "startswith", "kwargs": _OPERATOR_AUTO_ESCAPE},
        "endswith": {"method": "endswith", "kwargs": _OPERATOR_AUTO_ESCAPE},
        "contains": {"method": "contains", "kwargs": _OPERATOR_AUTO_ESCAPE},
    }

    #: Backend exceptions to convert to 422 "Unprocessable Entity"
    _BACKEND_EXCEPTIONS: dict[
        str, Callable[..., Type[BaseException] | tuple[Type[BaseException], ...]]
    ] = {"postgresql+asyncpg": _get_asyncpg_exception}

    def __init__(self, database: Database):
        """Initialize paginator.

        Args:
            database: Database instance.
        """
        self._database = database
        self._function_exceptions: Type[BaseException] | tuple[
            Type[BaseException], ...
        ] = self._BACKEND_EXCEPTIONS.get(database.url.scheme, _get_default_exception)()

    async def __call__(
        self,
        query: Select,
        obj_type: Type[BaseModelT],
        parameters: PageParameters,
    ) -> Page[BaseModelT]:
        """Select multiple rows from database and paginate result.

        Args:
            query: SELECT SQL query.
            obj_type: Object type to return.
            parameters: Pagination parameters.

        Returns:
            List of object.
        """
        since = parameters.since
        page = parameters.page
        limit = parameters.limit
        order_by = parameters.order_by
        filter_by = parameters.filter_by

        query = self._query_filter_by(query, filter_by, obj_type).distinct()
        alias = query.alias()

        main_col = query.selected_columns[0]
        main_field = main_col.name
        main_type = get_type_hints(obj_type)[main_field]
        query, main_descending = self._query_order_by(
            query, parameters.order_by, main_field
        )

        if since is not None:
            self._check_since_available(main_field, order_by)
            since = parse_obj_as(main_type, since)
            query = query.where(
                (main_col < since) if main_descending else (main_col > since)
            )
            total_pages = None
            next_page = None

        elif page is not None and page > 1:
            query = query.offset((page - 1) * limit)
            total_pages = None
            next_page = page + 1

        else:
            with self._handle_paginate_error(filter_by):
                total_pages = (
                    await self._database.fetch_val(select(count()).select_from(alias))
                    // limit
                    + 1
                )
            next_page = 2 if 1 < total_pages else None

        with self._handle_paginate_error(filter_by):
            response = await self._database.fetch_all(query.limit(limit))
        if len(response) < limit:
            next_page = next_since = None
        elif page is None:
            next_since = parse_obj_as(main_type, getattr(response[-1], main_field))
        else:
            next_since = None
        return Page(
            next_since=next_since,
            next_page=next_page,
            total_pages=total_pages,
            items=[obj_type.from_orm(row) for row in response],
        )

    @contextmanager
    def _handle_paginate_error(
        self,
        filter_by: list[str] | None,
    ) -> Generator[None, None, None]:
        """Handle paginate query errors.

        Args:
            filter_by: Filters parameters.
        """
        try:
            yield
        except self._function_exceptions:
            if filter_by is not None:
                raise _UnprocessableEntity(
                    "Operator is incompatible with data type.",
                    loc=("query", "filter_by"),
                )
            raise

    @staticmethod
    def _check_since_available(since: str, order_by: list[str] | None) -> None:
        """Check if order by is compatible with since.

        Args:
            since: Field used with since.
            order_by: Order by.
        """
        if order_by is not None and (
            len(order_by) != 1 or order_by[0].lstrip("-") != since
        ):
            raise _UnprocessableEntity(
                (
                    '"since" can only be used if "order_by" is sorted by field'
                    f' "{since}" only.'
                ),
                loc=("query", "since"),
            )

    @staticmethod
    def _get_column(query: Select, field: str) -> Column:
        """Get a column by field name from a query.

        Args:
            query: Query.
            field: Field name.

        Returns:
            Column.
        """
        for column in query.selected_columns:
            if column.name == field:
                return column  # type: ignore
        raise _UnprocessableEntity(
            f'Invalid field name "{field}" in query parameter.', loc=("query",)
        )

    @classmethod
    def _query_order_by(
        cls, query: Select, order_by: list[str] | None, main_field: str
    ) -> tuple[Select, bool]:
        """Update a select query with order by parameter.

        Args:
            query: Query.
            order_by: Order by parameters.
            main_field: Main field name.

        Returns:
            Updated query and boolean that is True if the main column is sorted
            descending.
        """
        if order_by is None:
            return query, False

        seen = set()
        main_descending = False
        for param in order_by:
            if param.startswith("-"):
                field = param[1:]
                descending = True
            else:
                field = param
                descending = False
            if field in seen:
                raise _UnprocessableEntity(
                    f'Duplicated field name "{field}" in order by query parameters.',
                    loc=("query", "order_by"),
                )
            elif descending and field == main_field:
                main_descending = True
            seen.add(field)
            column = cls._get_column(query, field)
            query = query.order_by(column.desc() if descending else column)
        return query, main_descending

    @classmethod
    def _parse_value(cls, field_type: Type[T], value: str) -> T:
        """Raw parser.

        Args:
            value: Value to parse.
            field_type: Field data type.

        Returns:
            Parsed value.
        """
        try:
            return parse_raw_as(field_type, value, json_loads=cls.json_loads)
        except JSONDecodeError:
            raise ValueError(f'Unable to parse filter argument value "{value}".')

    @classmethod
    def _get_operator(cls, operator: str, fil_str: str) -> tuple[dict[str, Any], bool]:
        """Get operator detail from operator string.

        Args:
            operator: Operator.
            fil_str: Filter str.

        Returns:
            Operator details, True if operator need negation.
        """
        try:
            return cls._OPERATORS[operator], False
        except KeyError:
            if operator.startswith("!"):
                try:
                    return cls._OPERATORS[operator[1:]], True
                except KeyError:
                    pass
        raise _UnprocessableEntity(
            f'Invalid filter operator "{operator}".',
            loc=("query", "filter_by", fil_str),
        )

    @classmethod
    def _query_filter_by(
        cls, query: Select, filter_by: list[str] | None, obj_type: Type[BaseModelT]
    ) -> Select:
        """Update a select query with filters.

        Args:
            query: Query.
            filter_by: Filters parameters.

        Returns:
            Updated query.
        """
        type_hints = get_type_hints(obj_type)
        if filter_by is not None:
            for fil_str in filter_by:
                try:
                    col_str, ope_str, val_str = fil_str.strip().split(" ", 2)
                except ValueError:
                    raise _UnprocessableEntity(
                        "Unable to parse filter query parameter.",
                        loc=("query", "filter_by", fil_str),
                    )
                operator, negation = cls._get_operator(ope_str, fil_str)
                column = cls._get_column(query, col_str)
                try:
                    args = operator.get("parser", _parse_args)(
                        val_str.strip(),
                        operator.get(
                            "value_type",
                            type_hints.get(column.name) or column.type.python_type,
                        ),
                    )
                except (ValidationError, ValueError) as error:
                    raise _UnprocessableEntity(
                        error, loc=("query", "filter_by", fil_str)
                    )
                expr = getattr(column, operator["method"])(
                    *args, **operator.get("kwargs", {})
                )
                query = query.where(not_(expr) if negation else expr)  # type: ignore

        return query
