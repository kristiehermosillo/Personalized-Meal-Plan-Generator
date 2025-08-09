# recipe_db.py
from typing import TypedDict, List

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
    steps: List[str]

def R(
    name: str,
    cuisine: str,
    course: str,
    tags: List[str],
    cal: int,
    p: int, c: int, f: int,
    ings: List[Ingredient],
    steps: List[str]
) -> Recipe:
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
    # --- Breakfasts ---
    R("Greek Yogurt Parfait", "mediterranean", "breakfast",
      ["vegetarian"], 350, 24, 40, 10,
      [{"item":"greek yogurt","qty":1,"unit":"cup"},
       {"item":"berries","qty":1,"unit":"cup"},
       {"item":"granola","qty":0.5,"unit":"cup"}],
      steps=[
          "Spoon half the yogurt into a bowl or glass.",
          "Add half the berries and half the granola.",
          "Repeat layers and serve immediately."
      ]),
    R("Overnight Oats", "american", "breakfast",
      ["vegetarian"], 400, 16, 60, 12,
      [{"item":"rolled oats","qty":0.75,"unit":"cup"},
       {"item":"milk (or alt)","qty":1,"unit":"cup"},
       {"item":"chia seeds","qty":1,"unit":"tbsp"}],
      steps=[
          "Combine oats, milk, and chia in a jar.",
          "Stir, cover, and refrigerate at least 4 hours or overnight.",
          "Top with fruit or nuts before eating (optional)."
      ]),
    R("Tofu Scramble", "american", "breakfast",
      ["vegan","gluten-free","dairy-free","low-carb"], 300, 22, 12, 18,
      [{"item":"firm tofu","qty":200,"unit":"g"},
       {"item":"spinach","qty":2,"unit":"cups"},
       {"item":"turmeric","qty":0.5,"unit":"tsp"}],
      steps=[
          "Drain tofu and crumble into a skillet.",
          "Cook 3–4 min, add turmeric and salt/pepper.",
          "Stir in spinach and cook until wilted, 1–2 min."
      ]),
    R("Avocado Toast", "american", "breakfast",
      ["vegetarian"], 420, 12, 44, 20,
      [{"item":"bread","qty":2,"unit":"slices"},
       {"item":"avocado","qty":1,"unit":"pc"},
       {"item":"lemon","qty":0.25,"unit":"pc"}],
      steps=[
          "Toast the bread.",
          "Mash avocado with a squeeze of lemon, salt, and pepper.",
          "Spread on toast; add chili flakes or herbs if you like."
      ]),
    R("Protein Smoothie", "american", "breakfast",
      ["gluten-free","dairy-free"], 330, 28, 35, 8,
      [{"item":"protein powder","qty":1,"unit":"scoop"},
       {"item":"banana","qty":1,"unit":"pc"},
       {"item":"almond milk","qty":1.5,"unit":"cup"}],
      steps=[
          "Add all ingredients to a blender.",
          "Blend until smooth, 30–60 seconds.",
          "Serve immediately."
      ]),

    # --- Lunch ---
    R("Quinoa Chickpea Bowl", "mediterranean", "lunch",
      ["vegan","gluten-free","dairy-free"], 520, 20, 78, 14,
      [{"item":"quinoa","qty":0.75,"unit":"cup"},
       {"item":"chickpeas","qty":1,"unit":"cup"},
       {"item":"cucumber","qty":0.5,"unit":"pc"}],
      steps=[
          "Cook quinoa per package; fluff and cool slightly.",
          "Rinse and drain chickpeas; dice cucumber.",
          "Combine quinoa, chickpeas, and cucumber; dress with olive oil, lemon, salt."
      ]),
    R("Grilled Chicken Salad", "american", "lunch",
      ["gluten-free","dairy-free","low-carb"], 480, 42, 18, 22,
      [{"item":"chicken breast","qty":200,"unit":"g"},
       {"item":"mixed greens","qty":3,"unit":"cups"},
       {"item":"olive oil","qty":1,"unit":"tbsp"}],
      steps=[
          "Season chicken with salt/pepper; grill or pan-cook 5–7 min/side until done.",
          "Slice chicken. Toss greens with olive oil, vinegar, and salt.",
          "Top salad with sliced chicken; add veggies if desired."
      ]),
    R("Tuna Rice Bowl", "asian", "lunch",
      ["pescatarian","dairy-free"], 550, 35, 65, 14,
      [{"item":"tuna (canned)","qty":1,"unit":"can"},
       {"item":"rice","qty":1,"unit":"cup"},
       {"item":"soy sauce","qty":1,"unit":"tbsp"}],
      steps=[
          "Cook rice (or use leftover).",
          "Drain tuna; fluff with a fork.",
          "Serve tuna over rice; drizzle soy sauce; add sliced scallions if you have them."
      ]),
    R("Lentil Soup", "middle-eastern", "lunch",
      ["vegan","gluten-free","dairy-free"], 420, 22, 60, 8,
      [{"item":"red lentils","qty":1,"unit":"cup"},
       {"item":"carrot","qty":1,"unit":"pc"},
       {"item":"onion","qty":0.5,"unit":"pc"}],
      steps=[
          "Dice onion and carrot; sauté in a pot with a little oil 3–4 min.",
          "Add rinsed lentils and 4 cups water or broth.",
          "Simmer 20–25 min until soft; season with salt, pepper, cumin if you like."
      ]),
    R("Caprese Sandwich", "italian", "lunch",
      ["vegetarian"], 540, 22, 60, 22,
      [{"item":"bread","qty":2,"unit":"slices"},
       {"item":"tomato","qty":1,"unit":"pc"},
       {"item":"mozzarella","qty":100,"unit":"g"}],
      steps=[
          "Toast bread if desired.",
          "Layer sliced tomato and mozzarella; season with salt, pepper, and olive oil.",
          "Add basil if available; close and slice."
      ]),

    # --- Dinner ---
    R("Salmon with Veg", "mediterranean", "dinner",
      ["pescatarian","gluten-free","dairy-free","low-carb"], 640, 42, 28, 34,
      [{"item":"salmon","qty":200,"unit":"g"},
       {"item":"broccoli","qty":2,"unit":"cups"},
       {"item":"olive oil","qty":1,"unit":"tbsp"}],
      steps=[
          "Preheat oven to 400°F/200°C.",
          "Toss broccoli with half the oil, salt, pepper; spread on tray.",
          "Place salmon on tray, brush with remaining oil; bake 12–15 min until flaky."
      ]),
    R("Stir-Fry Tofu & Veg", "asian", "dinner",
      ["vegan","gluten-free","dairy-free","low-carb"], 520, 28, 40, 20,
      [{"item":"firm tofu","qty":250,"unit":"g"},
       {"item":"mixed vegetables","qty":3,"unit":"cups"},
       {"item":"rice","qty":0.5,"unit":"cup"}],
      steps=[
          "Cook rice (optional).",
          "Cube tofu; stir-fry in a hot pan with a little oil until golden.",
          "Add vegetables and a splash of soy sauce; cook 3–5 min until crisp-tender."
      ]),
    R("Turkey Chili", "american", "dinner",
      ["gluten-free","dairy-free","low-carb"], 600, 45, 45, 18,
      [{"item":"ground turkey","qty":300,"unit":"g"},
       {"item":"kidney beans","qty":1,"unit":"cup"},
       {"item":"tomato sauce","qty":1,"unit":"cup"}],
      steps=[
          "Brown turkey in a pot 5–6 min; season with salt and chili powder.",
          "Add beans and tomato sauce; simmer 15–20 min.",
          "Adjust seasoning; serve hot."
      ]),
    R("Chickpea Curry", "indian", "dinner",
      ["vegan","gluten-free","dairy-free"], 580, 22, 78, 16,
      [{"item":"chickpeas","qty":2,"unit":"cups"},
       {"item":"coconut milk","qty":0.5,"unit":"cup"},
       {"item":"rice","qty":0.75,"unit":"cup"}],
      steps=[
          "Cook rice.",
          "In a pan, warm a little oil; add curry powder or paste if you have it.",
          "Add chickpeas and coconut milk; simmer 8–10 min; salt to taste. Serve over rice."
      ]),
    R("Pasta Primavera", "italian", "dinner",
      ["vegetarian"], 620, 22, 90, 16,
      [{"item":"pasta","qty":90,"unit":"g"},
       {"item":"zucchini","qty":0.5,"unit":"pc"},
       {"item":"parmesan","qty":30,"unit":"g"}],
      steps=[
          "Cook pasta in salted water until al dente.",
          "Sauté sliced zucchini in olive oil; add garlic if you like.",
          "Toss pasta with zucchini and a splash of pasta water; finish with parmesan."
      ]),
    R("Veggie Fajitas", "mexican", "dinner",
      ["vegan","dairy-free"], 560, 18, 82, 16,
      [{"item":"tortillas","qty":3,"unit":"pc"},
       {"item":"bell pepper","qty":1,"unit":"pc"},
       {"item":"onion","qty":0.5,"unit":"pc"}],
      steps=[
          "Slice pepper and onion; sauté on high heat until charred-tender.",
          "Warm tortillas in a dry pan.",
          "Serve veggies in tortillas with salsa or lime."
      ]),

    # --- Any / flexible ---
    R("Egg Fried Rice", "asian", "any",
      ["vegetarian"], 540, 18, 78, 16,
      [{"item":"rice","qty":1,"unit":"cup"},
       {"item":"egg","qty":2,"unit":"pc"},
       {"item":"peas","qty":0.5,"unit":"cup"}],
      steps=[
          "Scramble eggs in a hot oiled pan; remove.",
          "Add rice and peas; stir-fry 2–3 min with soy sauce.",
          "Return eggs; toss and serve."
      ]),
    R("Buddha Bowl", "mediterranean", "any",
      ["vegan","gluten-free","dairy-free"], 590, 20, 84, 18,
      [{"item":"quinoa","qty":0.75,"unit":"cup"},
       {"item":"sweet potato","qty":1,"unit":"pc"},
       {"item":"chickpeas","qty":0.5,"unit":"cup"}],
      steps=[
          "Roast cubed sweet potato at 425°F/220°C for 20–25 min.",
          "Cook quinoa.",
          "Assemble bowl with quinoa, sweet potato, chickpeas; drizzle tahini/lemon if available."
      ]),
    R("Shrimp Pasta", "italian", "any",
      ["pescatarian"], 610, 34, 82, 16,
      [{"item":"pasta","qty":100,"unit":"g"},
       {"item":"shrimp","qty":200,"unit":"g"},
       {"item":"garlic","qty":2,"unit":"clove"}],
      steps=[
          "Cook pasta until al dente.",
          "Sauté garlic in olive oil; add shrimp and cook 2–3 min until pink.",
          "Toss shrimp with pasta and a splash of pasta water; season and serve."
      ]),
]
