import sqlite3
from pathlib import Path
from fastmcp import FastMCP

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "reviews.db"

mcp = FastMCP("Customer Feedback Tools")


def run_query(sql, params=()):
    """Run a read-only SQL query and return the rows as a list of dicts."""
    conn = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@mcp.tool()
def get_overview() -> dict:
    """Overall numbers for the whole dataset: total reviews, average star rating, and the percentage of customers who recommend the product."""
    return run_query("""
        SELECT COUNT(*) AS total_reviews,
               ROUND(AVG(rating), 2) AS avg_rating,
               ROUND(100.0 * AVG(recommended), 1) AS recommend_pct
        FROM reviews
    """)[0]


@mcp.tool()
def stats_by_department() -> list[dict]:
    """Review count, average rating and recommend percentage for each department, sorted from lowest recommend percentage to highest."""
    return run_query("""
        SELECT department,
               COUNT(*) AS reviews,
               ROUND(AVG(rating), 2) AS avg_rating,
               ROUND(100.0 * AVG(recommended), 1) AS recommend_pct
        FROM reviews
        WHERE department IS NOT NULL
        GROUP BY department
        ORDER BY recommend_pct ASC
    """)


@mcp.tool()
def stats_by_class(department: str = "") -> list[dict]:
    """Same numbers as stats_by_department, broken down by clothing class (for example Dresses or Knits). Pass a department name to filter, or leave empty for all. Classes with fewer than 30 reviews are left out because the numbers would be unreliable."""
    return run_query("""
        SELECT class_name,
               COUNT(*) AS reviews,
               ROUND(AVG(rating), 2) AS avg_rating,
               ROUND(100.0 * AVG(recommended), 1) AS recommend_pct
        FROM reviews
        WHERE class_name IS NOT NULL
          AND (? = '' OR department = ?)
        GROUP BY class_name
        HAVING COUNT(*) >= 30
        ORDER BY recommend_pct ASC
    """, (department, department))


@mcp.tool()
def keyword_impact(keyword: str) -> dict:
    """How reviews that mention a word or short phrase compare with all reviews: how many mention it, their average rating, and their recommend percentage. The text is matched exactly, as a substring, so 'small' and 'too small' give different results. Search for the exact wording the user asked about (for example 'too small', 'runs large', 'itchy')."""
    word = keyword.strip().lower()
    if not word:
        return {"error": "keyword must not be empty"}
    row = run_query("""
        SELECT COUNT(*) AS mentions,
               ROUND(AVG(rating), 2) AS avg_rating,
               ROUND(100.0 * AVG(recommended), 1) AS recommend_pct
        FROM reviews
        WHERE LOWER(review_text) LIKE ?
    """, (f"%{word}%",))[0]
    row["keyword"] = keyword
    return row


@mcp.tool()
def search_reviews(keyword: str, max_rating: int = 5, limit: int = 5) -> list[dict]:
    """Find example reviews that contain a keyword and have a star rating at or below max_rating. Returns at most 10 reviews with their review_id, rating and full text."""
    word = keyword.strip().lower()
    limit = max(1, min(limit, 10))
    return run_query("""
        SELECT review_id, rating, recommended, department, class_name, review_text
        FROM reviews
        WHERE LOWER(review_text) LIKE ? AND rating <= ?
        ORDER BY positive_feedback DESC
        LIMIT ?
    """, (f"%{word}%", max_rating, limit))


if __name__ == "__main__":
    mcp.run()