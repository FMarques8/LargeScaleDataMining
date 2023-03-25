from pyspark import SparkContext
from pyspark.sql import SparkSession
import argparse

from operator import add # equivalent to lambda x, y : x + y
from itertools import combinations

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--support", help = "support threshold", type = int, default = 1000, nargs = 1)
parser.add_argument("-l", "--lift", help = "lift threshold", type = float, default = 0.2, nargs = 1)
parser.add_argument("directory", help = "directory of file", type = str, nargs = 1)
args = parser.parse_args()

if __name__ == "__main__":
    
    # Assign variables from command line
    support_threshold = args.support
    lift_threshold = args.lift
    f_dir = args.directory[0]
    
    # Initialize Spark
    sc = SparkContext(appName="Assignment1 - Conditions")
    spark = SparkSession.builder.appName("Assignment1 - Conditions").getOrCreate()
    print(sc.uiWebUrl)

    # Read file
    file = spark.read.option('header','true').csv(f_dir)
    print('File structure:')
    file.printSchema()
    
    # Save codes and their description to a dictionary
    code_to_desc = dict(file.select('CODE', 'DESCRIPTION').distinct().collect())
    
    # Clean file
    patient_rdd = (file.select('PATIENT', 'CODE') 
                    .distinct()
                    .rdd.map(lambda row: (row[0], int(row[1]))) # convert DataFrame to RDD with records saved as tuples
                    .groupByKey() # group values (codes) by patient
                    .mapValues(tuple) # turn groupByKey iterable to tuple
                    .map(lambda item: (item[0], sorted(item[1], reverse = True))) # sort codes by decreasing order
                    )

    # k = 1 frequent k-sets RDD
    frequent_items = (patient_rdd.flatMap(lambda item: item[1]) # apply flatmap to the codes
                    .map(lambda item: (item, 1)) # convert each code to (code, 1)
                    .reduceByKey(add) # count codes
                    .filter(lambda x: x[1] >= support_threshold) # remove codes that are not frequent, accordingly to threshold
                    .sortBy(lambda x: -x[1]) # sort from most frequent to least frequent
                    )

    print('10 most frequent conditions')
    print(frequent_items.take(10))

    # converts (code, count) to code,  and transforms collected list to set for faster lookup
    frequent_items_set = set(frequent_items.map(lambda item: item[0]).collect()) 
    
    # Compare original file size to processed RDD
    print(f'\nNumber of records in original dataframe: {file.rdd.count()}')
    print(f'Number of records after removing duplicate diagnosis: {patient_rdd.count()}')
    
    # APRIORI ALGORITHM
    # k = 2
    
    frequent_pairs = (patient_rdd.flatMap(lambda item: [pair for pair in combinations(item[1] , 2) 
                if all(value in frequent_items_set for value in pair)]) # verifies if both elements in the pair are frequent
                .map(lambda item: (item, 1))
                .reduceByKey(add) # counts items
                .filter(lambda item: item[1] >= support_threshold) # removes non-frequent pairs
                .sortBy(lambda item: -item[1]) # sorts from most frequent to least
                )

    print("\n10 most frequent pairs (k = 2 sets)")
    most_frequent_pairs = frequent_pairs.take(10)
    print(most_frequent_pairs)

    frequent_pairs_set = set(frequent_pairs.map(lambda item: item[0]).collect())
    
    # k = 3
    frequent_triplets = (patient_rdd.flatMap(lambda item: [triplet for triplet in combinations(item[1], 3) 
                    if all(subset in frequent_pairs_set for subset in combinations(triplet, 2))])
                    # verifies if every subset of length 2 of the triplet is frequent;
                    # discards triplets with at least 1 subset that doesn't verify this
                    .map(lambda item: (item, 1)) # same as above
                    .reduceByKey(add)
                    .filter(lambda item: item[1] >= support_threshold)
                    .sortBy(lambda item: -item[1])
                    )

    print("\n10 most frequent triplets (k = 3 sets)")
    most_frequent_triplets = frequent_triplets.take(10)
    print(most_frequent_triplets)

    frequent_triplet_set = set(frequent_triplets.map(lambda item: item[0]).collect())
    
    # ASSOCIATION RULES
    # Count total number of each k-set
    
    total_singles = sum(frequent_items.map(lambda item: item[1]).collect())

    total_pairs = sum(frequent_pairs.map(lambda item: item[1]).collect())

    total_triplets = sum(frequent_triplets.map(lambda item: item[1]).collect())

    print(f'\nTotal single items: {total_singles}')
    print(f'Total pairs: {total_pairs}')
    print(f'Total triplets: {total_triplets}')
    
    def std_lift(lift, probX, probY, baskets):
        numerator = lift - (max([probX + probY - 1, 1/baskets])/ (probX * probY))
        denominator = (1/max([probX, probY])) - (max([probX + probY - 1, 1/baskets])/(probX * probY))
        return (numerator / denominator)
    
    # Compute probabilities of each item and pair being 'drawn'
    item_probability = frequent_items.map(lambda item: (item[0], item[1] / total_singles)).collect()
    pair_probability = frequent_pairs.map(lambda item: (item[0], item[1] / total_pairs)).collect()

    # Collect to lift to iterate over
    frequent_items_list = frequent_items.collect()
    frequent_pairs_list = frequent_pairs.collect()
    
    # X → Y
    # Confidence
    XY_confidence = frequent_pairs.flatMap(lambda pair: [(X, tuple(set(pair[0]) - set((X,)))[0], pair[1] / supportX) 
                                                        for X, supportX in frequent_items_list 
                                                        if X in pair[0]]
                                        )

    # Interest
    XY_interest = XY_confidence.flatMap(lambda pair: [((pair[0]), pair[1], pair[2] - probY) 
                                                    for Y, probY in item_probability 
                                                    if Y == pair[1]]
                                        )

    # Lift
    XY_lift = XY_confidence.flatMap(lambda pair: [((pair[0]), pair[1], pair[2] / probY) 
                                                for Y, probY in item_probability 
                                                if Y == pair[1]]
                                    )

    # Std Lift
    XY_stdLift = XY_lift.flatMap(lambda pair: [((pair[0]), pair[1], std_lift(pair[2], probX, probY, total_pairs)) 
                                            for X, probX in item_probability for Y, probY in item_probability 
                                            if X == pair[0] and Y == pair[1] and X != Y]
                                )
    
    # Join in 1 DF
    XY_results = (XY_confidence.toDF(['X', 'Y', 'confidence'])
                .join(XY_interest.toDF(['X', 'Y', 'interest']), on = ['X', 'Y'])
                .join(XY_lift.toDF(['X', 'Y', 'lift']), on = ['X', 'Y'])
                .join(XY_stdLift.toDF(['X', 'Y', 'std_lift']), on = ['X', 'Y'])
                )
    
    # Select std_lift column and filter using those values
    XY_results_filtered = XY_results.filter(XY_results.std_lift >= lift_threshold)

    # Sort in descending order
    XY_results_filtered = XY_results_filtered.sort(XY_results_filtered.std_lift.desc())

    XY_rules = XY_results_filtered.collect()

    print(XY_results_filtered.show(10))
    
    # Save to files
    # With codes
    print('Saving rules to file...')
    
    with open('XY_rules_with_codes.txt', 'w') as f:
        f.write(f"{'X':<15} {'->':<5} {'Y':<15}\t{'Std Lift':<20}\t{'Lift':<20}\t{'Confidence':<20}\t{'Interest':<20}\n")
        for rule in XY_rules:
            f.write(f"{rule['X']:<15} {'->':<5} {rule['Y']:<15}\t{rule['std_lift']:<20}\t{rule['lift']:<20}\t{rule['confidence']:<20}\t{rule['interest']:<20}\n")

    # With description
    with open('XY_rules_with_description.txt', 'w') as f:
        f.write(f"{'X':<80} {'->':<5} {'Y':<80}\t{'Std Lift':<20}\t{'Lift':<20}\t{'Confidence':<20}\t{'Interest':<20}\n")
        for rule in XY_rules:
            f.write(f"{code_to_desc[str(rule['X'])]:<80} {'->':<5} {(code_to_desc[str(rule['Y'])]):<80}\t{rule['std_lift']:<20}\t{rule['lift']:<20}\t{rule['confidence']:<20}\t{rule['interest']:<20}\n")
            
    # XY → Z
    # Confidence
    XYZ_confidence = frequent_triplets.flatMap(lambda item: [((X, Y), tuple(set(item[0]) - set((X, Y)))[0], item[1] / supportPair) 
                                            for (X, Y), supportPair in frequent_pairs_list 
                                            if X in item[0] and Y in item[0]]
                                            )

    # Interest
    XYZ_interest = XYZ_confidence.flatMap(lambda item: [(item[0], item[1], item[2] - probZ) 
                                        for Z, probZ in item_probability 
                                        if Z == item[1]]
                                        )

    # Lift
    XYZ_lift = XYZ_confidence.flatMap(lambda item: [(item[0], item[1], item[2] / probZ) 
                                    for Z, probZ in item_probability 
                                    if Z == item[1]]
                                    )

    # Std Lift
    XYZ_stdLift = XYZ_lift.flatMap(lambda item: [(item[0], item[1], std_lift(item[2], probPair, probZ, total_triplets)) 
                                    for (X, Y), probPair in pair_probability for Z, probZ in item_probability 
                                    if X in item[0] and Y in item[0] and Z == item[1] and X != Z and Y != Z]
                                    )
    
    # Convert RDDs to DataFrames and join them on columns 'X' and 'Y'
    XYZ_results = (XYZ_confidence.toDF(['(X, Y)', 'Z', 'confidence'])
                .join(XYZ_interest.toDF(['(X, Y)', 'Z', 'interest']), on = ['(X, Y)', 'Z'])
                .join(XYZ_lift.toDF(['(X, Y)', 'Z', 'lift']), on = ['(X, Y)', 'Z'])
                .join(XYZ_stdLift.toDF(['(X, Y)', 'Z', 'std_lift']), on = ['(X, Y)', 'Z'])
                )

    # Select std_lift column and filter using those values
    XYZ_results_filtered = XYZ_results.filter(XYZ_results.std_lift >= lift_threshold)

    # Sort in descending order
    XYZ_results_filtered = XYZ_results_filtered.sort(XYZ_results_filtered.std_lift.desc())

    XYZ_rules = XYZ_results_filtered.collect()

    print(XYZ_results_filtered.show(10))
    
    print('Saving rules to file...')
    
    with open('XYZ_rules_with_codes.txt', 'w') as f:
        f.write(f"{'X, Y':<40} {'->':<5} {'Z':<15}\t{'Std Lift':<20}\t{'Lift':<20}\t{'Confidence':<20}\t{'Interest':<20}\n")
        for rule in XYZ_rules:
            X, Y = rule['(X, Y)']
            f.write(f"{X:<20}{Y:<20} {'->':<5} {rule['Z']:<15}\t{rule['std_lift']:<20}\t{rule['lift']:<20}\t{rule['confidence']:<20}\t{rule['interest']:<20}\n")
    
    with open('XYZ_rules_with_description.txt', 'w') as f:
        f.write(f"{'X, Y':<180} {'->':<5} {'Z':<80}\t{'Std Lift':<20}\t{'Lift':<20}\t{'Confidence':<20}\t{'Interest':<20}\n")
        for rule in XYZ_rules:
            X, Y = rule['(X, Y)']
            f.write(f"{code_to_desc[str(X)]:<90}{code_to_desc[str(Y)]:<90} {'->':<5} {(code_to_desc[str(rule['Z'])]):<80}\t{rule['std_lift']:<20}\t{rule['lift']:<20}\t{rule['confidence']:<20}\t{rule['interest']:<20}\n")
    
    # Save most frequent itemsets for k = 1,2,3
    with open('most_frequent_itemsets.txt', 'w') as f:
        # k = 1
        f.write(f"10 most frequent items for k = 1:\n{'Code':<42} {'Conditions':<120} {'Count':<15}\n")
        for item in frequent_items.take(10):
            f.write(f'{item[0]:<42} {code_to_desc[str(item[0])]:<120} {item[1]:<15}\n')
        # k = 2
        f.write(f"\n10 most frequent items for k = 2:\n{'Code':<42} {'Conditions':<120} {'Count':<15}\n")
        for pair in most_frequent_pairs:
            f.write(f'{pair[0][0]:<14}{pair[0][1]:<28} {code_to_desc[str(pair[0][0])]:<60}{code_to_desc[str(pair[0][1])]:<60} {pair[1]:<15}\n')
        # k = 3
        f.write(f"\n10 most frequent items for k = 3:\n{'Code':<42} {'Conditions':<120} {'Count':<15}\n")
        for triplet in most_frequent_triplets:
            f.write(f"""{triplet[0][0]:<14}{triplet[0][1]:<14}{triplet[0][1]:<14} {code_to_desc[str(triplet[0][0])]:<40}{code_to_desc[str(triplet[0][1])]:<40}{code_to_desc[str(triplet[0][2])]:<40} {triplet[1]:<15}\n""")