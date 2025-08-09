# recipe_db.py
from typing import TypedDict, List, Dict

class Macro(TypedDict):
    protein_g: int
    carbs_g: int
    fat_g: int

class Ingredient(TypedDict, total=False):
    item: str
    qty: float
    unit: str

class Recipe(TypedDict, total=False):
    name: str
    cuisine: str
    course: str         # breakfast | lunch | dinner | any
    diet_tags: List[str]
    calories: int
    macros: Macro
    ingredients: List[Ingredient]

def R(name, cuisine, course, tags, cal, p, c, f, ings, steps):
    return {
        "name": name,
        "cuisine": cuisine,
        "course": course,
        "diet_tags": tags,
        "calories": cal,
        "macros": {"protein_g": p, "carbs_g": c, "fat_g": f},
        "ingredients": ings,
        "steps": steps,
    }


RECIPE_DB: List[Recipe] = [
    # Breakfasts
    R("Greek Yogurt Parfait", "mediterranean", "breakfast",
      ["vegetarian"], 350, 24, 40, 10,
      [{"item":"greek yogurt","qty":1,"unit":"cup"}, {"item":"berries","qty":1,"unit":"cup"}, {"item":"granola","qty":0.5,"unit":"cup"}]),
    R("Overnight Oats", "american", "breakfast",
      ["vegetarian"], 400, 16, 60, 12,
      [{"item":"rolled oats","qty":0.75,"unit":"cup"}, {"item":"milk (or alt)","qty":1,"unit":"cup"}, {"item":"chia seeds","qty":1,"unit":"tbsp"}]),
    R("Tofu Scramble", "american", "breakfast",
      ["vegan","gluten-free","dairy-free","low-carb"], 300, 22, 12, 18,
      [{"item":"firm tofu","qty":200,"unit":"g"}, {"item":"spinach","qty":2,"unit":"cups"}, {"item":"turmeric","qty":0.5,"unit":"tsp"}]),
    R("Avocado Toast", "american", "breakfast",
      ["vegetarian"], 420, 12, 44, 20,
      [{"item":"bread","qty":2,"unit":"slices"}, {"item":"avocado","qty":1,"unit":"pc"}, {"item":"lemon","qty":0.25,"unit":"pc"}]),
    R("Protein Smoothie", "american", "breakfast",
      ["gluten-free","dairy-free"], 330, 28, 35, 8,
      [{"item":"protein powder","qty":1,"unit":"scoop"},
       {"item":"banana","qty":1,"unit":"pc"},
       {"item":"almond milk","qty":1.5,"unit":"cup"}],
      steps=[
          "Add all ingredients to a blender.",
          "Blend until smooth.",
          "Serve immediately."
  ])
# Lunch
    R("Quinoa Chickpea Bowl", "mediterranean", "lunch",
      ["vegan","gluten-free","dairy-free"], 520, 20, 78, 14,
      [{"item":"quinoa","qty":0.75,"unit":"cup"}, {"item":"chickpeas","qty":1,"unit":"cup"}, {"item":"cucumber","qty":0.5,"unit":"pc"}]),
    R("Grilled Chicken Salad", "american", "lunch",
      ["gluten-free","dairy-free","low-carb"], 480, 42, 18, 22,
      [{"item":"chicken breast","qty":200,"unit":"g"}, {"item":"mixed greens","qty":3,"unit":"cups"}, {"item":"olive oil","qty":1,"unit":"tbsp"}]),
    R("Tuna Rice Bowl", "asian", "lunch",
      ["pescatarian","dairy-free"], 550, 35, 65, 14,
      [{"item":"tuna (canned)","qty":1,"unit":"can"}, {"item":"rice","qty":1,"unit":"cup"}, {"item":"soy sauce","qty":1,"unit":"tbsp"}]),
    R("Lentil Soup", "middle-eastern", "lunch",
      ["vegan","gluten-free","dairy-free"], 420, 22, 60, 8,
      [{"item":"red lentils","qty":1,"unit":"cup"}, {"item":"carrot","qty":1,"unit":"pc"}, {"item":"onion","qty":0.5,"unit":"pc"}]),
    R("Caprese Sandwich", "italian", "lunch",
      ["vegetarian"], 540, 22, 60, 22,
      [{"item":"bread","qty":2,"unit":"slices"}, {"item":"tomato","qty":1,"unit":"pc"}, {"item":"mozzarella","qty":100,"unit":"g"}]),
    # Dinner
    R("Salmon with Veg", "mediterranean", "dinner",
      ["pescatarian","gluten-free","dairy-free","low-carb"], 640, 42, 28, 34,
      [{"item":"salmon","qty":200,"unit":"g"}, {"item":"broccoli","qty":2,"unit":"cups"}, {"item":"olive oil","qty":1,"unit":"tbsp"}]),
    R("Stir-Fry Tofu & Veg", "asian", "dinner",
      ["vegan","gluten-free","dairy-free","low-carb"], 520, 28, 40, 20,
      [{"item":"firm tofu","qty":250,"unit":"g"}, {"item":"mixed vegetables","qty":3,"unit":"cups"}, {"item":"rice","qty":0.5,"unit":"cup"}]),
    R("Turkey Chili", "american", "dinner",
      ["gluten-free","dairy-free","low-carb"], 600, 45, 45, 18,
      [{"item":"ground turkey","qty":300,"unit":"g"}, {"item":"kidney beans","qty":1,"unit":"cup"}, {"item":"tomato sauce","qty":1,"unit":"cup"}]),
    R("Chickpea Curry", "indian", "dinner",
      ["vegan","gluten-free","dairy-free"], 580, 22, 78, 16,
      [{"item":"chickpeas","qty":2,"unit":"cups"}, {"item":"coconut milk","qty":0.5,"unit":"cup"}, {"item":"rice","qty":0.75,"unit":"cup"}]),
    R("Pasta Primavera", "italian", "dinner",
      ["vegetarian"], 620, 22, 90, 16,
      [{"item":"pasta","qty":90,"unit":"g"}, {"item":"zucchini","qty":0.5,"unit":"pc"}, {"item":"parmesan","qty":30,"unit":"g"}]),
    R("Veggie Fajitas", "mexican", "dinner",
      ["vegan","dairy-free"], 560, 18, 82, 16,
      [{"item":"tortillas","qty":3,"unit":"pc"}, {"item":"bell pepper","qty":1,"unit":"pc"}, {"item":"onion","qty":0.5,"unit":"pc"}]),
    # Any / flexible
    R("Egg Fried Rice", "asian", "any",
      ["vegetarian"], 540, 18, 78, 16,
      [{"item":"rice","qty":1,"unit":"cup"}, {"item":"egg","qty":2,"unit":"pc"}, {"item":"peas","qty":0.5,"unit":"cup"}]),
    R("Buddha Bowl", "mediterranean", "any",
      ["vegan","gluten-free","dairy-free"], 590, 20, 84, 18,
      [{"item":"quinoa","qty":0.75,"unit":"cup"}, {"item":"sweet potato","qty":1,"unit":"pc"}, {"item":"chickpeas","qty":0.5,"unit":"cup"}]),
    R("Shrimp Pasta", "italian", "any",
      ["pescatarian"], 610, 34, 82, 16,
      [{"item":"pasta","qty":100,"unit":"g"}, {"item":"shrimp","qty":200,"unit":"g"}, {"item":"garlic","qty":2,"unit":"clove"}]),
]
