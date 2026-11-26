from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import re


def norm_text(s: Any) -> str:
    s = str(s).casefold().strip()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".!?,;: ").strip()

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "cyan", "teal", "magenta",
          "violet", "gold", "lime", "turquoise", "indigo"]
TODO_ITEMS = ["buy milk", "call the plumber", "water the plants", "pick up the dry cleaning", "renew the car insurance",
              "book a dentist appointment", "take out the recycling", "order printer ink", "clean the gutters",
              "pay the electricity bill", "replace the smoke detector battery", "buy birthday candles",
              "schedule the boiler service", "return the library books", "wash the car", "buy light bulbs",
              "send the rent cheque", "feed the neighbour's cat", "defrost the freezer", "fix the garden gate",
              "pick up the parcel", "buy dog food", "book the vet", "mow the lawn", "change the bed sheets",
              "restock the pantry", "refill the prescription", "buy stamps", "back up the laptop", "sort the garage"]
OOD_REQUESTS = [
    "what is the weather like in Paris tomorrow", "tell me a joke", "who won the football match last night",
    "what is the capital of Australia", "translate good morning into Spanish", "how many ounces are in a pound",
    "write me a short poem about autumn", "what is twelve times fifteen", "give me a recipe for pancakes",
    "how do I remove a coffee stain from a shirt", "set an alarm for seven in the morning", "call my mother",
    "play some jazz on my phone", "order a pizza", "what is the stock price of Apple", "read me the news headlines",
    "how far is the moon", "remind me to buy flowers on Friday", "send an email to my boss", "book a taxi to the airport",
    "what year did the Berlin Wall fall", "explain how photosynthesis works", "convert 30 celsius to fahrenheit",
    "who wrote Pride and Prejudice", "what should I cook tonight", "how tall is Mount Everest",
    "search the web for cheap flights", "what is the meaning of the word ephemeral", "add two plus two",
    "tell me about the history of Rome", "text my sister that I am late", "what time does the pharmacy close",
    "recommend a good book", "how do I tie a tie", "what is the speed of light", "sing me a song",
    "what is the best way to learn guitar", "how many days until Christmas", "turn the page", "open my calendar",
]
TRAIN_OOD_REQUESTS = [
    "what is the population of Canada", "how do I bake sourdough bread", "who painted the Mona Lisa",
    "tell me a fun fact about octopuses", "what is the square root of 144", "translate thank you into French",
    "recommend a movie for tonight", "how many calories are in an apple", "what does GDP stand for",
    "explain what a black hole is", "write a haiku about the sea", "who is the president of France",
    "what is the tallest building in the world", "how long does it take to boil an egg", "give me a workout plan",
    "what is the boiling point of water", "how do I change a flat tire", "find me a hotel in Lisbon",
    "what is the distance from Earth to Mars", "schedule a meeting with my team", "open the browser",
    "what is the plural of cactus", "how do vaccines work", "what is a good name for a puppy",
    "who invented the telephone", "what is the exchange rate for euros", "play my morning playlist",
    "summarize the plot of Hamlet", "what are the symptoms of a cold", "how do I make cold brew coffee",
    "what is the currency of Japan", "show me directions to the library", "how much is a gallon in liters",
    "who discovered penicillin", "what is machine learning", "where is the nearest gas station",
    "what is the largest ocean", "teach me a magic trick", "how many legs does a spider have", "what is inflation",
]
UNSUPPORTED = ["dishwasher", "washing machine", "oven", "coffee maker", "hot tub", "3D printer", "pool pump",
               "doorbell camera", "air purifier", "towel warmer", "robot lawn mower", "toaster", "clothes dryer",
               "electric kettle", "sauna", "projector", "garage heater", "water softener"]

PREFIX = ["", "", "", "please ", "hey, ", "go ahead and ", "I need you to ", "I want you to "]
SUFFIX = ["", "", "", " please", ".", " now"]


def _the(name: str) -> str:
    return "" if norm_text(name).startswith("the ") else "the "


def _form(rng: random.Random, name: str) -> str:
    return name if rng.random() < 0.5 else name.lower()


def wrap(rng: random.Random, clause: str) -> str:
    if rng.random() < 0.12:
        return "would you " + clause + "?"
    return rng.choice(PREFIX) + clause + rng.choice(SUFFIX)


DOMAIN_MIN = {"light": 6, "switch": 2, "fan": 1, "lock": 2, "cover": 3, "media_player": 3, "vacuum": 1,
              "timer": 2, "todo": 1, "climate": 1}


def _conflict(a: str, b: str) -> bool:
    a, b = norm_text(a), norm_text(b)
    return a in b or b in a


def make_house(catalog: List[Dict[str, str]], rng: random.Random, size: Optional[int] = None) -> List[Dict[str, str]]:
    size = size or rng.randint(26, 46)
    by_dom: Dict[str, List[Dict[str, str]]] = {}
    for d in catalog:
        by_dom.setdefault(d["entity_id"].split(".")[0], []).append(d)
    chosen: List[Dict[str, str]] = []

    def add(d: Dict[str, str]) -> bool:
        if any(_conflict(d["name"], c["name"]) or d["entity_id"] == c["entity_id"] for c in chosen):
            return False
        chosen.append(d)
        return True

    for dom, k in DOMAIN_MIN.items():
        pool = list(by_dom.get(dom, []))
        rng.shuffle(pool)
        got = 0
        for d in pool:
            if got >= k:
                break
            got += add(d)
    rest = list(catalog)
    rng.shuffle(rest)
    for d in rest:
        if len(chosen) >= size:
            break
        add(d)
    rng.shuffle(chosen)
    return chosen


def _pick(rng: random.Random, house: List[Dict[str, str]], domains: Tuple[str, ...], used: set) -> Optional[Dict[str, str]]:
    pool = [d for d in house if d["entity_id"].split(".")[0] in domains and d["entity_id"] not in used]
    return rng.choice(pool) if pool else None


Clause = Tuple[str, Dict[str, Any], str, str]


def c_onoff(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("light", "switch", "fan"), used)
    if not d:
        return None
    n = _form(rng, d["name"])
    t = _the(d["name"])
    kind = rng.choice(["on", "on", "off", "off", "toggle"])
    if kind == "on":
        clause = rng.choice([f"turn on {t}{n}", f"switch on {t}{n}", f"turn {t}{n} on", f"power up {t}{n}", f"activate {t}{n}"])
        tool = "HassTurnOn"
    elif kind == "off":
        clause = rng.choice([f"turn off {t}{n}", f"switch off {t}{n}", f"turn {t}{n} off", f"shut off {t}{n}", f"deactivate {t}{n}"])
        tool = "HassTurnOff"
    else:
        clause = rng.choice([f"toggle {t}{n}", f"flip {t}{n}"])
        tool = "HassToggle"
    return clause, {"name": tool, "arguments": {"name": d["name"]}}, d["entity_id"], "syn_onoff"


def c_lockcover(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("lock", "cover"), used)
    if not d:
        return None
    n = _form(rng, d["name"])
    t = _the(d["name"])
    if d["entity_id"].startswith("lock."):
        if rng.random() < 0.5:
            clause, tool = rng.choice([f"lock {t}{n}", f"secure {t}{n}"]), "HassTurnOff"
        else:
            clause, tool = f"unlock {t}{n}", "HassTurnOn"
    else:
        if rng.random() < 0.5:
            clause, tool = rng.choice([f"open {t}{n}", f"open up {t}{n}"]), "HassTurnOn"
        else:
            clause, tool = rng.choice([f"close {t}{n}", f"shut {t}{n}"]), "HassTurnOff"
    return clause, {"name": tool, "arguments": {"name": d["name"]}}, d["entity_id"], "syn_lockcover"


def c_brightness(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("light",), used)
    if not d:
        return None
    p = rng.randint(5, 100)
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"set the brightness of {t}{n} to {p} percent", f"dim {t}{n} to {p}%",
                         f"set {t}{n} to {p} percent brightness", f"change {t}{n} brightness to {p}%"])
    return clause, {"name": "HassLightSet", "arguments": {"name": d["name"], "brightness": p}}, d["entity_id"], "syn_brightness"


def c_color(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("light",), used)
    if not d:
        return None
    col = rng.choice(COLORS)
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"make {t}{n} {col}", f"change {t}{n} to {col}", f"set the color of {t}{n} to {col}", f"turn {t}{n} {col}"])
    return clause, {"name": "HassLightSet", "arguments": {"name": d["name"], "color": col}}, d["entity_id"], "syn_color"


def c_color_brightness(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("light",), used)
    if not d:
        return None
    col, p = rng.choice(COLORS), rng.randint(10, 100)
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"set {t}{n} to {col} at {p} percent brightness", f"make {t}{n} {col} with {p}% brightness"])
    return clause, {"name": "HassLightSet", "arguments": {"name": d["name"], "color": col, "brightness": p}}, d["entity_id"], "syn_color_brightness"


def c_position(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("cover",), used)
    if not d:
        return None
    p = rng.choice(range(0, 101, 5))
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"set {t}{n} to {p} percent open", f"move {t}{n} to position {p}", f"open {t}{n} to {p}%"])
    return clause, {"name": "HassSetPosition", "arguments": {"name": d["name"], "position": p}}, d["entity_id"], "syn_position"


def c_volume(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("media_player",), used)
    if not d:
        return None
    p = rng.choice(range(5, 101, 5))
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"set the volume of {t}{n} to {p} percent", f"set {t}{n} volume to {p}%", f"turn {t}{n} to {p} percent volume"])
    return clause, {"name": "HassSetVolume", "arguments": {"name": d["name"], "volume_level": round(p / 100, 2)}}, d["entity_id"], "syn_volume"


def c_media(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("media_player",), used)
    if not d:
        return None
    n, t = _form(rng, d["name"]), _the(d["name"])
    kind = rng.choice(["pause", "resume", "next"])
    if kind == "pause":
        clause, tool = rng.choice([f"pause {t}{n}", f"pause the music on {t}{n}"]), "HassMediaPause"
    elif kind == "resume":
        clause, tool = rng.choice([f"resume playback on {t}{n}", f"unpause {t}{n}"]), "HassMediaUnpause"
    else:
        clause, tool = rng.choice([f"skip to the next track on {t}{n}", f"play the next song on {t}{n}"]), "HassMediaNext"
    return clause, {"name": tool, "arguments": {"name": d["name"]}}, d["entity_id"], "syn_media"


def c_vacuum(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("vacuum",), used)
    if not d:
        return None
    n, t = _form(rng, d["name"]), _the(d["name"])
    if rng.random() < 0.5:
        clause, tool = rng.choice([f"start {t}{n}", f"send {t}{n} out to clean"]), "HassVacuumStart"
    else:
        clause, tool = rng.choice([f"send {t}{n} back to its dock", f"return {t}{n} to base"]), "HassVacuumReturnToBase"
    return clause, {"name": tool, "arguments": {"name": d["name"]}}, d["entity_id"], "syn_vacuum"


def c_todo(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("todo",), used)
    if not d:
        return None
    item = rng.choice(TODO_ITEMS)
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"add {item} to {t}{n}", f"put {item} on {t}{n}", f"add the task {item} to {t}{n}"])
    return clause, {"name": "HassListAddItem", "arguments": {"name": d["name"], "item": item}}, d["entity_id"], "syn_todo"


def c_timer(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("timer",), used)
    if not d:
        return None
    n, t = _form(rng, d["name"]), _the(d["name"])
    kind = rng.choice(["cancel", "pause", "resume"])
    if kind == "cancel":
        clause, tool = rng.choice([f"cancel {t}{n}", f"stop and cancel {t}{n}"]), "HassCancelTimer"
    elif kind == "pause":
        clause, tool = f"pause {t}{n}", "HassPauseTimer"
    else:
        clause, tool = rng.choice([f"resume {t}{n}", f"unpause {t}{n}"]), "HassUnpauseTimer"
    return clause, {"name": tool, "arguments": {"name": d["name"]}}, d["entity_id"], "syn_timer"


def c_climate(rng, house, used) -> Optional[Clause]:
    d = _pick(rng, house, ("climate",), used)
    if not d:
        return None
    temp: Any = rng.randint(16, 28)
    if rng.random() < 0.25:
        temp = temp + 0.5
    n, t = _form(rng, d["name"]), _the(d["name"])
    clause = rng.choice([f"set {t}{n} to {temp} degrees", f"set the temperature on {t}{n} to {temp}", f"change the target temperature of {t}{n} to {temp} degrees"])
    return clause, {"name": "HassClimateSetTemperature", "arguments": {"name": d["name"], "temperature": temp}}, d["entity_id"], "syn_climate"


SINGLE_GENS: Dict[str, Callable] = {
    "syn_onoff": c_onoff, "syn_lockcover": c_lockcover, "syn_brightness": c_brightness, "syn_color": c_color,
    "syn_color_brightness": c_color_brightness, "syn_position": c_position, "syn_volume": c_volume,
    "syn_media": c_media, "syn_vacuum": c_vacuum, "syn_todo": c_todo, "syn_timer": c_timer, "syn_climate": c_climate,
}
MULTI_GENS = [c_onoff, c_lockcover, c_brightness, c_color, c_media, c_vacuum]


def gen_single(rng: random.Random, house: List[Dict[str, str]], category: str) -> Optional[Dict[str, Any]]:
    c = SINGLE_GENS[category](rng, house, set())
    if not c:
        return None
    clause, gold, _eid, cat = c
    return {"request": wrap(rng, clause), "gold_calls": [gold], "category": cat}


def gen_multi(rng: random.Random, house: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    k = rng.choice([2, 2, 3])
    used: set = set()
    clauses, golds = [], []
    gens = rng.sample(MULTI_GENS, k=min(k + 1, len(MULTI_GENS)))
    for g in gens:
        if len(clauses) == k:
            break
        c = g(rng, house, used)
        if c:
            clauses.append(c[0])
            golds.append(c[1])
            used.add(c[2])
    if len(clauses) < 2:
        return None
    if len(clauses) == 2:
        text = f"{clauses[0]} {rng.choice(['and', 'and then'])} {clauses[1]}"
    else:
        text = f"{clauses[0]}, {clauses[1]} and {clauses[2]}"
    return {"request": wrap(rng, text), "gold_calls": golds, "category": "syn_multi"}


def gen_nocall(rng: random.Random, house: List[Dict[str, str]], category: str,
               ood_pool: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    if category == "nocall_ood":
        return {"request": rng.choice(ood_pool or OOD_REQUESTS) + rng.choice(["", "", "?", " please"]), "gold_calls": [], "category": category}
    if category == "nocall_unsupported":
        words = [w for w in UNSUPPORTED if not any(w.lower() in d["name"].lower() for d in house)]
        w = rng.choice(words)
        verb = rng.choice(["turn on", "start", "switch off", "turn off"])
        return {"request": f"{verb} the {w}", "gold_calls": [], "category": category}
    if category == "nocall_state":
        d = _pick(rng, house, ("light", "switch", "fan", "lock", "cover"), set())
        n, t = _form(rng, d["name"]), _the(d["name"])
        dom = d["entity_id"].split(".")[0]
        q = {"lock": [f"is {t}{n} locked", f"is {t}{n} unlocked"], "cover": [f"is {t}{n} open", f"is {t}{n} closed"]}.get(
            dom, [f"is {t}{n} on", f"is {t}{n} off", f"what is the state of {t}{n}"])
        return {"request": rng.choice(q) + rng.choice(["?", ""]), "gold_calls": [], "category": category}
    raise KeyError(category)


def generate(catalog: List[Dict[str, str]], seed: int, quotas: Dict[str, int], taken: set, unique: bool = True,
             ood_pool: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    for cat, quota in sorted(quotas.items()):
        made, guard = 0, 0
        while made < quota and guard < quota * 200:
            guard += 1
            house = make_house(catalog, rng)
            if cat == "syn_multi":
                r = gen_multi(rng, house)
            elif cat.startswith("nocall_"):
                r = gen_nocall(rng, house, cat, ood_pool)
            else:
                r = gen_single(rng, house, cat)
            if not r:
                continue
            key = norm_text(r["request"])
            if key in taken:
                continue
            if unique:
                taken.add(key)
            r.update({"source": "synthetic", "devices": house})
            items.append(r)
            made += 1
        if made < quota:
            raise RuntimeError(f"failed to generate {quota} items for {cat}")
    return items
