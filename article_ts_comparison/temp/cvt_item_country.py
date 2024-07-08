from stdlib.tprint import tprint


def main():
    tprint("Start ...")
    nlines = 0
    with open("data/tb_food_import_features_month.csv", mode='r') as fin:
        with open("data/tb_food_import_features_month_item_country.csv", mode="w") as fout:
            for line in fin:
                line = line.replace('","', '~')
                fout.writelines([line])
                nlines += 1
                tprint(f"... {nlines}", force=False)
    tprint(f"Done {nlines}")
# end with/with/for


if __name__ == "__main__":
    main()
