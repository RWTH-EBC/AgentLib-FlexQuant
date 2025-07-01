from datetime import datetime, timedelta

def get_day_of_year(year: int, day_of_year: int):
    """
    Retrieve the date and weekday for a given year and day of the year.

    Args:
        year (int): The year (e.g., 2015).
        day_of_year (int): The day of the year (e.g., 87).

    Returns:
        tuple: A tuple containing the date (YYYY-MM-DD) and the weekday name.
    """
    # Calculate the date
    start_of_year = datetime(year, 1, 1)
    target_date = start_of_year + timedelta(days=day_of_year - 1)

    # Get the weekday name
    weekday_name = target_date.strftime("%A")
    print(f"Year: {year}, Day of Year: {day_of_year}, Date: {target_date.strftime('%Y-%m-%d')}, Weekday: {weekday_name}")

    return target_date.strftime("%Y-%m-%d"), weekday_name