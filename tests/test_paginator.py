"""Tests."""
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from typing import Any
from databases import Database
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    LargeBinary,
    DateTime,
)
from webuuid import Uuid


class Client(TestClient):
    """Test client."""

    def get_paginated(
        self, url: str, item_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get all items from a paginated result.

        Args:
            url: URL.
            item_id: If specified, filter result to find only the object with this ID.

        Returns:
            Items.
        """
        items = list()
        params = dict(limit="100")
        params_sep = "&" if "?" in url else "?"
        if item_id is not None:
            params["filter_by"] = quote(f'id = "{item_id}"')

        while True:
            params_str = "&".join(f"{key}={value}" for key, value in params.items())
            response = self.request("GET", f"{url}{params_sep}{params_str}")
            assert response.status_code == 200, response.text
            details = response.json()
            params["since"] = details.get("next_since")
            items += details["items"]

            if params["since"] is None or (item_id is not None and items):
                break

        if item_id is not None:
            assert len(items) <= 1
            assert all(item["id"] == item_id for item in items), [
                item["id"] for item in items
            ]
        return items


def test_paginator_sqlite(tmp_path: Path) -> None:
    """Test paginator with SQlite."""
    from fastapi_paginator import Paginator, Page, PageParameters

    # Initialize test database
    metadata = MetaData()
    table = Table(
        "table",
        metadata,
        Column("id", LargeBinary(length=16), primary_key=True, nullable=False),
        Column("index", Integer, nullable=False),
        Column("name", String, nullable=False),
        Column("label", String),
        Column("quantity", Integer),
        Column("date", DateTime(), nullable=False),
    )
    database_url = f"sqlite:///{tmp_path.joinpath('local.db')}"
    metadata.create_all(create_engine(database_url))
    database = Database(database_url)
    paginator = Paginator(database)

    # Initialize test application
    app = FastAPI()
    client = Client(app)

    class Item(BaseModel):
        """Item in database."""

        class Config:
            """Config."""

            orm_mode = True
            json_encoders = {Uuid: str}

        id: Uuid = Field(default_factory=Uuid)
        index: int
        name: str
        label: str | None
        quantity: int | None
        date: datetime = Field(default_factory=datetime.now)

    @app.post("/add", status_code=201)
    async def add_data(item: Item) -> None:
        """Add data."""
        await database.execute(table.insert().values(**item.dict()))

    @app.get("/list")
    async def list_data(
        page_parameters: PageParameters = Depends(),
    ) -> Page[Item]:
        """List data."""
        return await paginator(table.select(), Item, page_parameters)

    # Create some initial data
    count = 64
    for i in range(count):
        response = client.post(
            "/add",
            json=dict(
                name=f"Name {i}",
                index=count - i,
                label=(
                    "".join(
                        (
                            "G" if i % 2 else "M",
                            "E" if i % 4 else "I",
                            "V" if i % 3 else "B",
                        )
                    )
                    if i % 5
                    else None
                ),
                quantity=i % 10 + 1,
            ),
        )
        assert response.status_code == 201, response.text

    # Test navigation by "page"
    response = client.get(
        # Page 1, implicit
        "/list?order_by=id"
    )
    assert response.status_code == 200, response.text
    details = response.json()
    assert details["next_page"] == 2
    assert details["total_pages"] == 4
    assert details["next_since"]
    seen = set(item["id"] for item in details["items"])
    assert len(seen) == 20

    response = client.get(
        # Page 1, explicit
        "/list?order_by=id&page=1"
    )
    assert response.status_code == 200, response.text
    details = response.json()
    assert details["next_page"] == 2
    assert details["total_pages"] == 4
    assert not details.get("next_since")
    page_ids = set(item["id"] for item in details["items"])
    assert len(page_ids) == 20
    assert all(uuid in seen for uuid in page_ids)

    for i in range(2, 4):
        response = client.get(
            # Other pages
            f"/list?order_by=id&page={i}"
        )
        assert response.status_code == 200, response.text
        details = response.json()
        assert details["next_page"] == i + 1
        assert not details.get("total_pages")
        assert not details.get("next_since")
        page_ids = set(item["id"] for item in details["items"])
        assert len(page_ids) == 20
        assert all(uuid not in seen for uuid in page_ids), [
            uuid in seen for uuid in page_ids
        ]
        seen.update(page_ids)

    response = client.get(
        # Last page
        "/list?order_by=id&page=4"
    )
    assert response.status_code == 200, response.text
    details = response.json()
    assert not details.get("next_page")
    assert not details.get("total_pages")
    assert not details.get("next_since")
    page_ids = set(item["id"] for item in details["items"])
    assert len(page_ids) == 4
    assert all(uuid not in seen for uuid in page_ids)

    response = client.get(
        # Page > maximum pages
        "/list?order_by=id&page=5"
    )
    assert response.status_code == 200, response.text
    details = response.json()
    assert not details.get("next_page")
    assert not details.get("total_pages")
    assert not details.get("next_since")
    assert len(details["items"]) == 0

    response = client.get(
        # Out of range page
        "/list?order_by=id&page=0"
    )
    assert response.status_code == 422, response.text

    # Test navigation by "since"
    response = client.get(
        # First page
        "/list"
    )
    assert response.status_code == 200, response.text
    details = response.json()
    seen = set(item["id"] for item in details["items"])
    assert len(seen) == 20
    next_since = details.get("next_since")
    assert next_since
    assert next_since in seen

    for _ in range(2):
        response = client.get(
            # Other pages
            f"/list?since={next_since}"
        )
        assert response.status_code == 200, response.text
        details = response.json()
        assert not details.get("next_page")

        page_ids = set(item["id"] for item in details["items"])
        assert len(page_ids) == 20
        assert next_since not in page_ids

        next_since = details.get("next_since")
        assert next_since
        assert next_since in page_ids

        assert all(tuple(uuid not in seen for uuid in page_ids))
        seen.update(page_ids)

    response = client.get(
        # Last page
        f"/list?since={next_since}"
    )
    assert response.status_code == 200, response.text
    details = response.json()
    assert not details.get("next_page")
    assert not details.get("next_since")
    assert len(details["items"]) == 4
    page_ids = set(item["id"] for item in details["items"])
    assert len(page_ids) == 4
    assert all(uuid not in seen for uuid in page_ids)

    # Test using both "page" and "since" in the same request
    # TODO: Returns 500 here (ValidationError not handled properly ?)
    # response = client.get(f"/list?page=1&since={next_since}")
    # assert response.status_code in 422, response.text

    # Test "limit"
    response = client.get(
        # limit greater that the number of values
        "/list?limit=100"
    )
    assert response.status_code == 200, response.text
    all_items = response.json()["items"]
    assert len(all_items) == count

    response = client.get(
        # limit shorter that the number of values
        "/list?limit=42"
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == 42

    response = client.get(
        # Out of range limit
        "/list?limit=200"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Out of range limit
        "/list?limit=0"
    )
    assert response.status_code == 422, response.text

    # Test "order_by"
    response = client.get(
        # Sort ascending
        "/list?order_by=index"
    )
    assert response.status_code == 200, response.text
    items = [item["index"] for item in response.json()["items"]]
    assert items == sorted(items)

    response = client.get(
        # Sort descending
        "/list?order_by=-index"
    )
    assert response.status_code == 200, response.text
    items = [item["index"] for item in response.json()["items"]]
    assert items == sorted(items, reverse=True)

    response = client.get(
        # Sort by multiple columns
        "/list?order_by=quantity&order_by=index"
    )
    assert response.status_code == 200, response.text
    items = [(item["quantity"], item["index"]) for item in response.json()["items"]]
    assert items == sorted(items)

    response = client.get(
        # "since" used with order_by only by the same column
        f"/list?order_by=-id&since={next_since}"
    )
    assert response.status_code == 200, response.text

    response = client.get(
        # "since" used when order_by another column
        f"/list?order_by=index&since={next_since}"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Same column used multiple time
        "/list?order_by=index.asc&order_by=index.desc"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Bad asc/desc indication
        "/list?order_by=index.ASC"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Invalid column
        "/list?order_by=not_exists"
    )
    assert response.status_code == 422, response.text

    # Test "filter_by"
    items = client.get_paginated(
        # Operator ">" with int column
        f"/list?filter_by={quote('quantity > 5')}"
    )
    assert items
    assert all(item["quantity"] > 5 for item in items)

    items = client.get_paginated(
        # Operator ">=" with int column
        f"/list?filter_by={quote('quantity >= 5')}"
    )
    assert items
    assert all(item["quantity"] >= 5 for item in items)

    items = client.get_paginated(
        # Operator "<" with int column
        f"/list?filter_by={quote('quantity < 5')}"
    )
    assert items
    assert all(item["quantity"] < 5 for item in items)

    items = client.get_paginated(
        # Operator "<=" with int column
        f"/list?filter_by={quote('quantity <= 5')}"
    )
    assert items
    assert all(item["quantity"] <= 5 for item in items)

    items = client.get_paginated(
        # Operator "=" with int column
        f"/list?filter_by={quote('quantity = 5')}"
    )
    assert items
    assert all(item["quantity"] == 5 for item in items)

    items = client.get_paginated(
        # Operator "=" with UUID column
        "/list?filter_by="
        + quote(f'id = "{all_items[0]["id"]}"')
    )
    assert items
    assert all(item["id"] == all_items[0]["id"] for item in items)

    items = client.get_paginated(
        # Operator "!=" with int column
        f"/list?filter_by={quote('quantity != 5')}"
    )
    assert items
    assert all(item["quantity"] != 5 for item in items)

    items = client.get_paginated(
        # Operator "in" with int column
        f"/list?filter_by={quote('quantity in 1,5,7')}"
    )
    assert items
    assert all(item["quantity"] in (1, 5, 7) for item in items)

    items = client.get_paginated(
        # Operator "!in" with int column
        f"/list?filter_by={quote('quantity !in 1,5,7')}"
    )
    assert items
    assert all(item["quantity"] not in (1, 5, 7) for item in items)

    items = client.get_paginated(
        # Operator "between" with int column
        f"/list?filter_by={quote('quantity between 2,6')}"
    )
    assert items
    assert all(2 <= item["quantity"] <= 6 for item in items)

    items = client.get_paginated(
        # Operator combination on a same column
        f"/list?filter_by={quote('quantity > 2')}&filter_by={quote('quantity < 8')}"
    )
    assert items
    assert all(2 < item["quantity"] < 8 for item in items)

    items = client.get_paginated(
        # Operator "startswith" with str column
        "/list?filter_by="
        + quote('label startswith "M"')
    )
    assert items
    assert all(item["label"].startswith("M") for item in items)

    items = client.get_paginated(
        # Operator "!startswith" with str column
        "/list?filter_by="
        + quote('label !startswith "M"')
    )
    assert items
    assert all(item["label"].startswith("G") for item in items)

    items = client.get_paginated(
        # Operator "endswith" with str column
        "/list?filter_by="
        + quote('label endswith "B"')
    )
    assert items
    assert all(item["label"].endswith("B") for item in items)

    items = client.get_paginated(
        # Operator "!endswith" with str column
        "/list?filter_by="
        + quote('label !endswith "B"')
    )
    assert items
    assert all(item["label"].endswith("V") for item in items)

    items = client.get_paginated(
        # Operator "endswith" with str column
        "/list?filter_by="
        + quote('label endswith "B"')
    )
    assert items
    assert all(item["label"].endswith("B") for item in items)

    items = client.get_paginated(
        # Operator "like" with str column
        "/list?filter_by="
        + quote('label like "MI%"')
    )
    assert items
    assert all(item["label"].startswith("MI") for item in items)

    response = client.get(
        # Operator "like" with int column
        "/list?filter_by="
        + quote('quantity like "2%"')
    )
    assert response.status_code == 200, response.text

    items = client.get_paginated(
        # Operator "!like" with str column
        "/list?filter_by="
        + quote('label !like "MI%"')
    )
    assert items
    assert all(not item["label"].startswith("MI") for item in items)

    items = client.get_paginated(
        # Operator "ilike" with str column
        "/list?filter_by="
        + quote('label ilike "mi%"')
    )
    assert items
    assert all(item["label"].startswith("MI") for item in items)

    items = client.get_paginated(
        # Operator "!ilike" with str column
        "/list?filter_by="
        + quote('label !ilike "mi%"')
    )
    assert items
    assert all(not item["label"].startswith("MI") for item in items)

    items = client.get_paginated(
        # Operator "contains" with str column
        "/list?filter_by="
        + quote('label contains "I"')
    )
    assert items
    assert all("I" in item["label"] for item in items)

    items = client.get_paginated(
        # Operator "!contains" with str column
        "/list?filter_by="
        + quote('label !contains "I"')
    )
    assert items
    assert all("I" not in item["label"] for item in items)

    response = client.get(
        # Missing quotes on str value
        "/list?filter_by=label contains I"
    )
    assert response.status_code == 422, response.text

    filter_str = f'date < "{datetime.now().isoformat()}"'
    assert client.get_paginated(
        # Datetime value
        f"/list?filter_by={quote(filter_str)}"
    )

    assert client.get_paginated(
        # Operator == with None value
        f"/list?filter_by={quote('quantity != null')}"
    )

    assert not client.get_paginated(
        # Operator == with None value
        f"/list?filter_by={quote('quantity = null')}"
    )

    response = client.get(
        # Invalid operator
        "/list?filter_by=quantity not_exist 1"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Invalid operator with negation
        "/list?filter_by=quantity !not_exist 1"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Invalid column
        "/list?filter_by=not_exist = 1"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Invalid value
        '/list?filter_by=quantity = "a"'
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Operator "between" with invalid number of args
        f"/list?filter_by={quote('quantity between 2,6,8')}"
    )
    assert response.status_code == 422, response.text

    response = client.get(
        # Invalid query
        "/list?filter_by=quantity"
    )
    assert response.status_code == 422, response.text
