from gurobipy import *
import pandas as pd
import sqlite3
from math import dist


def solve(database_path, json_path, objective):
    # read the NN-Vector clumns == pictures
    vector = pd.read_json(json_path)

    # read in the database file
    con = sqlite3.connect("./pd.db")

    styles = pd.read_sql("SELECT * FROM styles", con)
    styles = styles.set_index("id")
    t_style_categories = pd.read_sql("SELECT * FROM style_categories", con)
    categories = pd.read_sql("SELECT * FROM categories", con)
    categories = categories.set_index("id")
    colors = pd.read_sql("SELECT * FROM colors", con)
    colors = colors.set_index("id")
    shop_categories = pd.read_sql("SELECT * FROM shop_categories", con)
    shops = pd.read_sql("SELECT * FROM shops", con)
    shops = shops.set_index("id")

    con.close()

    style_categories = pd.DataFrame(index=styles.index, columns=["category_id"])

    for i in styles.index:
        style_categories.at[i, "category_id"] = t_style_categories["category_id"][
                                                    t_style_categories["style_id"] == i].values[:]

    min_delivery = pd.DataFrame(0, index=shops.index, columns=categories.index)
    max_delivery = pd.DataFrame(0, index=shops.index, columns=categories.index)
    for s in shops.index:
        for i in categories.index:
            min_delivery.at[s, i] = shop_categories["min_delivery"][
                (shop_categories["shop_id"] == s) & (shop_categories["category_id"] == i)].values[0]
            max_delivery.at[s, i] = shop_categories["max_delivery"][
                (shop_categories["shop_id"] == s) & (shop_categories["category_id"] == i)].values[0]

    distance = {}
    for i in styles.index:
        for j in styles.index:
            distance[i, j] = dist(vector.loc[:, i].values[:], vector.loc[:, j].values[:])

    model = Model("pd")
    model.modelSense = GRB.MAXIMIZE

    # add Variables
    x = {}
    for s in shops.index:
        for i in styles.index:
            x[s, i] = model.addVar(name="x_%s_%s" % (s, i), vtype="I", lb=0)

    v = {}
    for s in shops.index:
        v[s] = model.addVar(name="v_%s" % (s), vtype="C")

    I = {}  # y in paper
    for s in shops.index:
        for i in styles.index:
            I[s, i] = model.addVar(name="I_%s_%s" % (s, i), vtype="B")

    y = {}  #
    for s in shops.index:
        for i in styles.index:
            for j in styles.index:
                y[s, i, j] = model.addVar(name="y_%s_%s,%s" % (s, i, j), vtype="B")

    w = {}
    for s in shops.index:
        for i in styles.index:
            for j in styles.index:
                if j > i:
                    w[s, i, j] = model.addVar(name="w_%s_%s,%s" % (s, i, j), vtype="C", lb=0)

    u = {}
    for s in shops.index:
        for i in styles.index:
            u[s, i] = model.addVar(name="u_%s_%s" % (s, i), vtype="C", lb=0)

    r = {}
    for s in shops.index:
        r[s] = model.addVar(name="u_%s_%s" % (s, i), vtype="C", lb=0)

    model.update()

    # add constraints

    # max/min delivery constraint
    for s in shops.index:
        for j in categories.index:
            model.addConstr(float(max_delivery.at[s, j]) >= quicksum(
                x[s, i] for i in styles.index if j in style_categories.at[i, "category_id"]))

            model.addConstr(float(min_delivery.at[s, j]) <= quicksum(
                x[s, i] for i in styles.index if j in style_categories.at[i, "category_id"]))

    # min shipment of style constraint
    for s in shops.index:
        for i in styles.index:
            model.addConstr(x[s, i] >= styles.at[i, "min_shipment"] * I[s, i])

            # supply constraint
    for i in styles.index:
        model.addConstr(quicksum(x[s, i] for s in shops.index) <= styles.at[i, "supply"])

    # min/max color percentage constraint at each store
    for s in shops.index:
        for j in colors.index:
            # max color percentage
            model.addConstr(quicksum(x[s, i] for i in styles.index if styles.at[i, "color_id"] == j) <= colors.at[
                j, "max_percentage"] * quicksum(x[s, i] for i in styles.index))

            # min color percentage
            model.addConstr(quicksum(x[s, i] for i in styles.index if styles.at[i, "color_id"] == j) >= colors.at[
                j, "min_percentage"] * quicksum(x[s, i] for i in styles.index))

    # variety measure validation
    for s in shops.index:
        model.addConstr(quicksum(I[s, i] for i in styles.index) >= 2)

    # Linking I and y (MaxSumSum)
    for s in shops.index:
        for i in styles.index:
            for j in styles.index:
                model.addConstr(I[s, i] <= y[s, i, j])
                model.addConstr(I[s, j] <= y[s, i, j])

    # MaxMean Constrs
    for s in shops.index:
        for i in styles.index:
            model.addConstr(u[s, i] >= r[s] + I[s, i] - 1)
            model.addConstr(u[s, i] <= I[s, i])

    for s in shops.index:
        model.addConstr(quicksum(u[s, i] for i in styles.index) == 1)

    for s in shops.index:
        for i in styles.index:
            for j in styles.index:
                if j > i:
                    model.addConstr(w[s, i, j] >= r[s] + I[s, i] + I[s, j] - 2)

                    model.addConstr(w[s, i, j] <= I[s, i])

                    model.addConstr(w[s, i, j] <= I[s, j])

                    model.addConstr(w[s, i, j] <= r[s])

    # MaxSumSum oder MaxMean Constraint (Betrag von I[s] fehlt noch)
    if objective == "MaxSumSum":
        for s in shops.index:
            model.addConstr(
                v[s] == quicksum(distance[i, j] * y[s, i, j] for i in styles.index for j in styles.index if i < j))

    elif objective == "MaxMean":
        for s in shops.index:
            model.addConstr(
                v[s] == quicksum(distance[i, j] * w[s, i, j] for i in styles.index for j in styles.index if i < j))

    # Objective function
    model.setObjective(quicksum(v[s] for s in shops.index))

    model.update()
    model.optimize()

    # model.computeIIS()
    model.write("model.lp")


solve("./pd.db", "./image2vec.json", "MaxSumSum")
