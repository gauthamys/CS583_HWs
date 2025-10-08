from collections import defaultdict
from itertools import combinations
import sys

class MSApriori:
    def __init__(self):
        self.transactions = []
        self.mis_values = {}
        self.prices = {}
        self.sdc = 0.0
        self.avpt = 0.0
        self.total_transactions = 0
        self.frequent_itemsets = {}
        self.item_support_cache = {}  # Cache for 1-item supports

    def read_data_file(self, filename):
        """Read transaction data from file"""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        transaction = [int(x.strip()) for x in line.split(',') if x.strip()]
                        if transaction:
                            self.transactions.append(set(transaction))
            self.total_transactions = len(self.transactions)
            print(f"Read {self.total_transactions} transactions")
        except FileNotFoundError:
            print(f"Error: Could not find data file '{filename}'")
            sys.exit(1)

    def read_data_file(self, filename):
        """Read transaction data from file"""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        transaction = [int(x.strip('\ufeff').strip()) for x in line.split(',') if x.strip()]
                        if transaction:
                            self.transactions.append(set(transaction))
            self.total_transactions = len(self.transactions)
            print(f"Read {self.total_transactions} transactions")
        except FileNotFoundError:
            print(f"Error: Could not find data file '{filename}'")
            sys.exit(1)
    
    def read_parameter_file(self, filename):
        """Read MIS values, prices, SDC, and AVPT from parameter file"""
        self.mis_rest = 0.01
        self.price_rest = 5
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('MIS('):
                        parts = line.split('=')
                        if len(parts) == 2:
                            item_part = parts[0].strip()
                            value = float(parts[1].strip())
                            
                            if 'rest' in item_part:
                                self.mis_rest = value
                            else:
                                item = int(item_part.split('(')[1].split(')')[0])
                                self.mis_values[item] = value
                    
                    elif line.startswith('Price(') or line.startswith('Prince('):
                        parts = line.split('=')
                        if len(parts) == 2:
                            item_part = parts[0].strip()
                            value = float(parts[1].strip())
                            
                            if 'rest' in item_part:
                                self.price_rest = value
                            else:
                                item = int(item_part.split('(')[1].split(')')[0])
                                self.prices[item] = value
                    
                    elif line.startswith('SDC'):
                        self.sdc = float(line.split('=')[1].strip())
                    
                    elif line.startswith('AVPT'):
                        self.avpt = float(line.split('=')[1].strip())
            
            print(f"Read parameters: SDC={self.sdc}, AVPT={self.avpt}")
        
        except FileNotFoundError:
            print(f"Error: Could not find parameter file '{filename}'")
            sys.exit(1)
    
    def get_mis(self, item):
        """Get MIS value for an item"""
        return self.mis_values.get(item, self.mis_rest)
    
    def get_price(self, item):
        """Get price for an item"""
        return self.prices.get(item, self.price_rest)
    
    def get_support_count(self, itemset):
        """Calculate support count for an itemset with optimized counting"""
        count = 0
        itemset_tuple = tuple(sorted(itemset))
        
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count
    
    def get_tail_count(self, itemset):
        """
        Calculate tail count - transactions containing all items except the last one
        For sorted itemset [i1, i2, ..., ik], tail count is count of transactions 
        containing [i1, i2, ..., i(k-1)]
        """
        if len(itemset) <= 1:
            return self.total_transactions
        
        sorted_items = sorted(list(itemset))
        prefix = set(sorted_items[:-1])
        
        count = 0
        for transaction in self.transactions:
            if prefix.issubset(transaction):
                count += 1
        return count
    
    def get_average_price(self, itemset):
        """Calculate average price of items in itemset"""
        total_price = sum(self.get_price(item) for item in itemset)
        return total_price / len(itemset)
    
    def satisfies_sdc(self, itemset):
        """Check if itemset satisfies Support Difference Constraint"""
        if len(itemset) <= 1:
            return True
        
        supports = []
        for item in itemset:
            # Use cached support if available
            if item in self.item_support_cache:
                supports.append(self.item_support_cache[item])
            else:
                support = self.get_support_count({item}) / self.total_transactions
                self.item_support_cache[item] = support
                supports.append(support)
        
        max_support = max(supports)
        min_support = min(supports)
        
        return (max_support - min_support) <= self.sdc
    
    def satisfies_avpt(self, itemset):
        """Check if itemset satisfies Average Price Threshold"""
        avg_price = self.get_average_price(itemset)
        return avg_price >= self.avpt
    
    def init_pass(self):
        """Initial pass to find frequent 1-itemsets using MSApriori logic"""
        # Count individual items
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Cache support values
        for item, count in item_counts.items():
            self.item_support_cache[item] = count / self.total_transactions
        
        # Sort items by MIS values (ascending), then by item number
        all_items = sorted(item_counts.keys(), key=lambda x: (self.get_mis(x), x))
        
        # Find M = first item with support >= MIS(item)
        M = None
        for item in all_items:
            if self.item_support_cache[item] >= self.get_mis(item):
                M = item
                break
        
        if M is None:
            return []
        
        # L is the set of frequent 1-itemsets
        frequent_1_itemsets = []
        
        for item in all_items:
            itemset = {item}
            support_count = item_counts[item]
            
            # Item must have support >= MIS(M) (not its own MIS for items after M)
            # This is the key difference in MSApriori
            if item == M:
                min_support = self.get_mis(M)
            else:
                # For items after M in sorted order, use MIS(M)
                min_support = self.get_mis(M)
            
            support = self.item_support_cache[item]
            
            # Check if item meets support and AVPT requirements
            if support >= min_support and self.satisfies_avpt(itemset):
                tail_count = self.get_tail_count(itemset)
                avg_price = self.get_average_price(itemset)
                frequent_1_itemsets.append((itemset, support_count, tail_count, avg_price))
        
        return frequent_1_itemsets
    
    def level2_candidate_gen(self, frequent_1):
        """Generate level-2 candidates using MSApriori-specific logic"""
        candidates = []
        items_list = [list(itemset)[0] for itemset, _, _, _ in frequent_1]
        
        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                item_i = items_list[i]
                item_j = items_list[j]
                
                # Check if support(item_i) >= MIS(item_i)
                if self.item_support_cache[item_i] >= self.get_mis(item_i):
                    candidate = {item_i, item_j}
                    candidates.append(candidate)
        
        return candidates
    
    def msapriori_candidate_gen(self, frequent_prev, k):
        """Generate candidates of size k using MSApriori join and prune steps"""
        candidates = []
        itemsets = [(frozenset(itemset), count, tc, ap) for itemset, count, tc, ap in frequent_prev]
        
        # Join step: join itemsets that differ by exactly one item
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                set_i = set(itemsets[i][0])
                set_j = set(itemsets[j][0])
                
                # Create sorted lists
                list_i = sorted(list(set_i))
                list_j = sorted(list(set_j))
                
                # Check if first k-2 items are the same
                if list_i[:-1] == list_j[:-1]:
                    # Create candidate by union
                    candidate = set_i.union(set_j)
                    
                    if len(candidate) == k:
                        candidates.append(candidate)
        
        # Prune step: remove candidates with infrequent subsets
        pruned_candidates = []
        frequent_sets = set(itemsets[i][0] for i in range(len(itemsets)))
        
        for candidate in candidates:
            sorted_items = sorted(list(candidate))
            c1 = sorted_items[0]  # First item
            
            # Check all (k-1)-subsets
            all_subsets_frequent = True
            
            for item in candidate:
                subset = candidate - {item}
                
                # Special check: if subset contains c1 or MIS(item) = MIS(c1)
                if c1 in subset or self.get_mis(item) == self.get_mis(c1):
                    if frozenset(subset) not in frequent_sets:
                        all_subsets_frequent = False
                        break
            
            if all_subsets_frequent:
                pruned_candidates.append(candidate)
        
        return pruned_candidates
    
    def run_msapriori(self):
        """Main MSApriori algorithm with proper implementation"""
        print("Starting MSApriori algorithm...")
        
        # Initial pass for 1-itemsets
        frequent_1 = self.init_pass()
        if not frequent_1:
            print("No frequent 1-itemsets found")
            return
        
        self.frequent_itemsets[1] = frequent_1
        print(f"Found {len(frequent_1)} frequent 1-itemsets")
        
        # Generate 2-itemsets
        candidates_2 = self.level2_candidate_gen(frequent_1)
        frequent_2 = []
        
        for candidate in candidates_2:
            support_count = self.get_support_count(candidate)
            
            # Use minimum MIS of items in the candidate
            min_mis = min(self.get_mis(item) for item in candidate)
            support = support_count / self.total_transactions
            
            if (support >= min_mis and 
                self.satisfies_sdc(candidate) and 
                self.satisfies_avpt(candidate)):
                
                tail_count = self.get_tail_count(candidate)
                avg_price = self.get_average_price(candidate)
                frequent_2.append((candidate, support_count, tail_count, avg_price))
        
        if frequent_2:
            self.frequent_itemsets[2] = frequent_2
            print(f"Found {len(frequent_2)} frequent 2-itemsets")
        else:
            print("No frequent 2-itemsets found")
            return
        
        # Generate k-itemsets for k >= 3
        k = 3
        while True:
            candidates = self.msapriori_candidate_gen(self.frequent_itemsets[k-1], k)
            
            if not candidates:
                break
            
            frequent_k = []
            for candidate in candidates:
                support_count = self.get_support_count(candidate)
                
                min_mis = min(self.get_mis(item) for item in candidate)
                support = support_count / self.total_transactions
                
                if (support >= min_mis and 
                    self.satisfies_sdc(candidate) and 
                    self.satisfies_avpt(candidate)):
                    
                    tail_count = self.get_tail_count(candidate)
                    avg_price = self.get_average_price(candidate)
                    frequent_k.append((candidate, support_count, tail_count, avg_price))
            
            if not frequent_k:
                break
            
            self.frequent_itemsets[k] = frequent_k
            print(f"Found {len(frequent_k)} frequent {k}-itemsets")
            k += 1
        
        print("MSApriori algorithm completed")
    
    def write_output(self, filename):
        """Write results to output file in required format"""
        with open(filename, 'w') as f:
            for length in sorted(self.frequent_itemsets.keys()):
                itemsets = self.frequent_itemsets[length]
                f.write(f"(Length-{length} {len(itemsets)}\n")
                
                for itemset, freq_count, tail_count, avg_price in itemsets:
                    sorted_items = sorted(list(itemset))
                    itemset_str = ' '.join(map(str, sorted_items))
                    f.write(f"({itemset_str}) : {freq_count} : {tail_count} : {avg_price:.0f}\n")
                
                f.write(")\n")
        print(f"Results written to {filename}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python msapriori.py <data_file> <parameter_file> <output_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    parameter_file = sys.argv[2]
    output_file = sys.argv[3]
    
    ms_apriori = MSApriori()
    ms_apriori.read_data_file(data_file)
    ms_apriori.read_parameter_file(parameter_file)
    ms_apriori.run_msapriori()
    ms_apriori.write_output(output_file)

if __name__ == "__main__":
    main()