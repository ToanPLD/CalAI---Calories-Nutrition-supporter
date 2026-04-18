def build_text(p):
    return f"""
    {p['food_name']} contains {p['macros']['calories']} kcal,
    {p['macros']['protein']}g protein,
    {p['macros']['carbs']}g carbs,
    {p['macros']['fat']}g fat.

    Category: {p['context'].get('category')}
    """