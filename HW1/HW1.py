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
        
    def read_data_file(self, filename):
        """Read transaction data from file"""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Parse comma-separated integers
                        transaction = [int(x.strip()) for x in line.split(',') if x.strip()]
                        if transaction:  # Only add non-empty transactions
                            self.transactions.append(set(transaction))
            self.total_transactions = len(self.transactions)
            print(f"Read {self.total_transactions} transactions")
        except FileNotFoundError:
            print(f"Error: Could not find data file '{filename}'")
            sys.exit(1)
    
    def read_parameter_file(self, filename):
        """Read MIS values, prices, SDC, and AVPT from parameter file"""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('MIS('):
                        # Parse MIS(item) = value
                        parts = line.split('=')
                        if len(parts) == 2:
                            item_part = parts[0].strip()
                            value = float(parts[1].strip())
                            
                            if 'rest' in item_part:
                                self.mis_rest = value
                            else:
                                # Extract item number from MIS(item)
                                item = int(item_part.split('(')[1].split(')')[0])
                                self.mis_values[item] = value
                    
                    elif line.startswith('Price('):
                        # Parse Price(item) = value or Prince(item) = value (typo in example)
                        parts = line.split('=')
                        if len(parts) == 2:
                            item_part = parts[0].strip()
                            value = float(parts[1].strip())
                            
                            if 'rest' in item_part:
                                self.price_rest = value
                            else:
                                # Extract item number
                                item = int(item_part.split('(')[1].split(')')[0])
                                self.prices[item] = value
                    
                    elif line.startswith('SDC'):
                        # Parse SDC = value
                        self.sdc = float(line.split('=')[1].strip())
                    
                    elif line.startswith('AVPT'):
                        # Parse AVPT = value
                        self.avpt = float(line.split('=')[1].strip())
            
            print(f"Read parameters: SDC={self.sdc}, AVPT={self.avpt}")
            print(f"MIS values for specific items: {self.mis_values}")
            print(f"Prices for specific items: {self.prices}")
        
        except FileNotFoundError:
            print(f"Error: Could not find parameter file '{filename}'")
            sys.exit(1)
    
    def get_mis(self, item):
        """Get MIS value for an item"""
        return self.mis_values.get(item, getattr(self, 'mis_rest', 0.01))
    
    def get_price(self, item):
        """Get price for an item"""
        return self.prices.get(item, getattr(self, 'price_rest', 5))
    
    def get_support(self, itemset):
        """Calculate support count for an itemset"""
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count
    
    def get_tail_count(self, itemset):
        """Calculate tail count - transactions containing all items except the last one"""
        if len(itemset) <= 1:
            return self.total_transactions
        
        # Convert to sorted list to get "last" item
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
            item_support = self.get_support({item}) / self.total_transactions
            supports.append(item_support)
        
        max_support = max(supports)
        min_support = min(supports)
        
        return (max_support - min_support) <= self.sdc
    
    def satisfies_avpt(self, itemset):
        """Check if itemset satisfies Average Price Threshold"""
        avg_price = self.get_average_price(itemset)
        return avg_price >= self.avpt
    
    def init_pass(self):
        """Initial pass to find frequent 1-itemsets"""
        # Count individual items
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Find all unique items and sort by MIS values
        all_items = list(item_counts.keys())
        all_items.sort(key=lambda x: (self.get_mis(x), x))
        
        frequent_1_itemsets = []
        
        for item in all_items:
            itemset = {item}
            support_count = item_counts[item]
            support = support_count / self.total_transactions
            mis = self.get_mis(item)
            
            if support >= mis and self.satisfies_avpt(itemset):
                tail_count = self.get_tail_count(itemset)
                avg_price = self.get_average_price(itemset)
                frequent_1_itemsets.append((itemset, support_count, tail_count, avg_price))
        
        return frequent_1_itemsets
    
    def generate_candidates(self, frequent_itemsets, k):
        """Generate candidate itemsets of size k"""
        candidates = []
        itemsets = [itemset for itemset, _, _, _ in frequent_itemsets]
        
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Join itemsets
                union_set = itemsets[i].union(itemsets[j])
                if len(union_set) == k:
                    candidates.append(union_set)
        
        # Remove duplicates
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def prune_candidates(self, candidates, previous_frequent, k):
        """Prune candidates using Apriori property"""
        pruned = []
        
        for candidate in candidates:
            # Check if all (k-1)-subsets are frequent
            valid = True
            for item in candidate:
                subset = candidate - {item}
                if not any(subset == freq_set for freq_set, _, _, _ in previous_frequent):
                    valid = False
                    break
            
            if valid:
                pruned.append(candidate)
        
        return pruned
    
    def run_msapriori(self):
        """Main MSApriori algorithm"""
        print("Starting MSApriori algorithm...")
        
        # Initial pass for 1-itemsets
        frequent_1 = self.init_pass()
        self.frequent_itemsets[1] = frequent_1
        print(f"Found {len(frequent_1)} frequent 1-itemsets")
        
        k = 2
        while True:
            # Generate candidates
            candidates = self.generate_candidates(self.frequent_itemsets[k-1], k)
            if not candidates:
                break
            
            # Prune candidates
            if k > 2:
                candidates = self.prune_candidates(candidates, self.frequent_itemsets[k-1], k)
            
            # Find frequent k-itemsets
            frequent_k = []
            for candidate in candidates:
                support_count = self.get_support(candidate)
                
                # Check minimum MIS requirement
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
                    # Sort items in itemset for consistent output
                    sorted_items = sorted(list(itemset))
                    itemset_str = ' '.join(map(str, sorted_items))
                    f.write(f"({itemset_str}) : {freq_count} : {tail_count} : {avg_price:.0f}\n")
                
                f.write(")\n")

def main():
    if len(sys.argv) != 4:
        print("Usage: python msapriori.py <data_file> <parameter_file> <output_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    parameter_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Create MSApriori instance
    ms_apriori = MSApriori()
    
    # Read input files
    ms_apriori.read_data_file(data_file)
    ms_apriori.read_parameter_file(parameter_file)
    
    # Run algorithm
    ms_apriori.run_msapriori()
    
    # Write output
    ms_apriori.write_output(output_file)
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()
