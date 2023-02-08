"""Pydantic models."""
from typing import Any, TypeVar, Generic
from fastapi import Query
from pydantic import BaseModel, Field, root_validator
from pydantic.generics import GenericModel


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class PageParameters(BaseModel):
    """Model for query parameters.

    To add as argument in routes as follows:
    page_parameters: PageQueryParameters=Depends()
    """

    order_by: list[str] | None = Field(
        Query(
            None,
            max_length=64,
            regex=r"^-?[a-z_]+$",
            description="""
Sort the resulting items by the specified field name.

Order is descending if `-` is added before the field name, else order is ascending.

This query parameter can be specified multiple time to sort by multiple columns.
""",
            example=(
                "Ordering descending by the `created_at` column: `order_by=-created_at`"
            ),
        )
    )

    filter_by: list[str] | None = Field(
        Query(
            None,
            max_length=64,
            description="""
Filter the resulting items.

The query must be in the form `field_name operator argument`, with:
  * `field_name`:  the name on the field on where apply the filter.
  * `operator`:  one operator from the list bellow.
  * `argument`: is the operator argument, it can be one or more value separated by `,`
    (Depending on the operator), valid values must be a primitive JSON type like
    numbers, double-quoted strings, `true`, `false` and `null`.

This query parameter can be specified multiple time to filter on more criteria
(Using AND logical conjunction).

Available operators:
  * `=`: Equal to a single value (Also supports `null`, `true` and `false`)
  * `<`: Lower than a single value.
  * `<=`: Lower or equal than a single value.
  * `>`: Greater than a single value.
  * `>=`: Greater or equal than a single value.
  * `between`: Between a pair of values (`value_1` <= `field_value` <= `value_2`).
  * `in`: Present in a list of one or more values.
  * `like`: Like a single value (`%` can be used as wildcard for zero to multiple
     characters, `_` as wildcard for a single character, `/` can be used as escape
     character for `%` and `_`).
  * `ilike`: Same as `like`, but case insensitive.
  * `startswith`: String representation starts with a single value.
  * `endswith`: String representation ends with a single value.
  * `contains`: String representation contains a single value.

Any operator can be negated by adding `!` in front of it.

*Warning*: Depending on your HTTP client, the query parameter value may require to be
URL encoded.
""",
            example="""
Returning only data with a `name name` field that does not start with
`Product`: `filter_by=name%20%21like%20%22Product%25%22`
(With URL encoded value of: `name !like "Product%"`')
""",
        )
    )

    limit: int = Field(
        Query(20, ge=1, le=100, description="Maximum item count to return.")
    )

    page: int | None = Field(
        Query(
            None,
            ge=1,
            description="""
The page to return.

When page is not specified or equal to `1`, the request returns `total_page` that is
the maximum number of pages.

*Cannot be used with `since`*.
""",
        )
    )

    since: str | None = Field(
        Query(
            None,
            max_length=64,
            description="""
The item from where starting to return the result.

When navigating between successive pages, the `next_since` returned value should be used
as `since` for the subsequent requests.

*Cannot be used with `page`*.

*Cannot be used with `order_by` if not ordering on the field used by `since`*.
""",
        )
    )

    @root_validator
    def _validate_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters.

        Args:
            values: Parameters.

        Returns:
            Parameters
        """
        if values.get("page") and values.get("since"):
            raise ValueError('"since" and "page" parameters cannot be used together.')
        return values


_PageItemT = TypeVar("_PageItemT")


class Page(GenericModel, Generic[_PageItemT]):
    """Model for paginated responses."""

    # Returned items
    items: list[_PageItemT]

    # Next value to use with "since"
    next_since: Any | None

    # Next value to use with "page"
    next_page: int | None

    # Total pages, only computed and returned when on page 1
    total_pages: int | None
