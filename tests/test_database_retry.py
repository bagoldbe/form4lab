"""Covers the database retry / connection-check behavior."""


def test_check_db_connection_succeeds():
    """check_db_connection should not raise for a valid engine."""
    from form4lab.database import check_db_connection
    check_db_connection()  # should not raise


def test_pool_recycle_is_120():
    """Verify pool_recycle is 120 seconds."""
    from form4lab.database import engine
    assert engine.pool._recycle == 120
