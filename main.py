import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
import warnings
from web3 import Web3
import asyncio
import aiohttp
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WalletMetrics:
    """Enhanced data class to store wallet risk metrics"""
    total_borrowed: float = 0.0
    total_supplied: float = 0.0
    net_worth: float = 0.0
    liquidation_count: int = 0
    transaction_count: int = 0
    unique_tokens: int = 0
    avg_health_factor: float = 0.0
    volatility_score: float = 0.0
    large_transaction_ratio: float = 0.0
    recent_activity_ratio: float = 0.0
    compound_usage_duration: int = 0
    max_leverage_used: float = 0.0
    position_concentration: float = 0.0
    frequency_score: float = 0.0
    gas_efficiency: float = 0.0
    protocol_loyalty: float = 0.0

class EnhancedCompoundDataFetcher:
    """Enhanced fetcher with multiple data sources and better parsing"""
    
    def __init__(self, etherscan_api_key: str = None):
        # Compound V2 and V3 contract addresses (more comprehensive)
        self.compound_contracts = {
            # Compound V2
            '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b': 'Comptroller_V2',
            '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643': 'cDAI',
            '0x39aa39c021dfbae8fac545936693ac917d5e7563': 'cUSDC_V2', 
            '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9': 'cUSDT',
            '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5': 'cETH',
            '0xccf4429db6322d5c611ee964527d42e5d685dd6a': 'cWBTC_V2',
            '0x70e36f6bf80a52b3b46b3af8e106cc0ed743e8e4': 'cLEND',
            '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4': 'cWBTC_V1',
            '0x6c8c6b02e7b2be14d4fa6022dfd6d75921d90e4e': 'cBAT',
            '0x158079ee67fce2f58472a96584a73c7ab9ac95c1': 'cREP',
            
            # Compound V3
            '0xc3d688b66703497daa19211eedff47f25384cdc3': 'cUSDCv3',
            '0xa17581a9e3356d9a858b789d68b4d866e593ae94': 'cETHv3',
            '0x9c4ec768c28520b50860ea7a15bd7213a9ff58bf': 'cUSDTv3',
            
            # Additional V2 tokens
            '0x35a18000230da775cac24873d00ff85bccded550': 'cUNI',
            '0x4b0181102a0112a2ef11abee5563bb4a3176c9d7': 'cSUSHI',
            '0x95b4ef2869ebd94beb4eee400a99824bf5dc325b': 'cMKR',
            '0xface851a4921ce59e912d19329929ce6da6eb0c7': 'cLINK',
        }
        
        # Enhanced function signatures
        self.compound_functions = {
            '0xa9059cbb': 'transfer',
            '0x23b872dd': 'transferFrom', 
            '0x1249c58b': 'mint',
            '0xdb006a75': 'redeem',
            '0x852a12e3': 'redeemUnderlying',
            '0xc5ebeaec': 'borrow',
            '0x0e752702': 'repayBorrow',
            '0x4e4d9fea': 'repayBorrowBehalf',
            '0xaae40a2a': 'liquidateBorrow',
            '0x4576b5db': 'seize',
            '0x317b0b77': 'enterMarkets',
            '0xe9c714f2': 'exitMarket',
            # V3 specific functions
            '0x1cff79cd': 'supply',
            '0xf2fde38b': 'withdraw',
            '0x414bf389': 'absorb',
            '0xb7929c62': 'buyCollateral',
        }
        
        self.etherscan_api_key = etherscan_api_key
        self.etherscan_base = "https://api.etherscan.io/api"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.3 if etherscan_api_key else 1.0  # Slower for free tier
        
    def _rate_limit(self):
        """Enhanced rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_wallet_transactions(self, wallet_address: str) -> List[Dict]:
        """Enhanced transaction fetching with better filtering"""
        logger.info(f"Fetching transactions for wallet: {wallet_address}")
        
        all_transactions = []
        
        # Fetch normal transactions
        normal_txs = self._fetch_etherscan_transactions(wallet_address, 'txlist')
        if normal_txs:
            all_transactions.extend(normal_txs)
        
        # Fetch internal transactions
        internal_txs = self._fetch_etherscan_transactions(wallet_address, 'txlistinternal')
        if internal_txs:
            all_transactions.extend(internal_txs)
        
        # Fetch ERC20 token transfers
        token_txs = self._fetch_etherscan_transactions(wallet_address, 'tokentx')
        if token_txs:
            all_transactions.extend(token_txs)
        
        # Filter for Compound-related transactions with better logic
        compound_transactions = self._enhanced_compound_filter(all_transactions)
        
        logger.info(f"Found {len(compound_transactions)} Compound transactions for {wallet_address}")
        return compound_transactions
    
    def _fetch_etherscan_transactions(self, wallet_address: str, action: str) -> List[Dict]:
        """Fetch different types of transactions from Etherscan"""
        self._rate_limit()
        
        params = {
            'module': 'account',
            'action': action,
            'address': wallet_address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': 1000,
            'sort': 'desc'
        }
        
        if self.etherscan_api_key:
            params['apikey'] = self.etherscan_api_key
        
        try:
            response = requests.get(self.etherscan_base, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1' and data.get('result'):
                    return data.get('result', [])
                elif data.get('message') == 'No transactions found':
                    return []
                else:
                    logger.warning(f"API response for {wallet_address} ({action}): {data.get('message', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"Etherscan API error for {wallet_address} ({action}): {e}")
        
        return []
    
    def _enhanced_compound_filter(self, transactions: List[Dict]) -> List[Dict]:
        """Enhanced filtering with better logic"""
        compound_related = []
        compound_addresses = set(addr.lower() for addr in self.compound_contracts.keys())
        
        for tx in transactions:
            is_compound = False
            
            # Check direct interactions with Compound contracts
            to_addr = tx.get('to', '').lower()
            from_addr = tx.get('from', '').lower()
            contract_addr = tx.get('contractAddress', '').lower()
            
            if (to_addr in compound_addresses or 
                from_addr in compound_addresses or 
                contract_addr in compound_addresses):
                is_compound = True
            
            # Check function signatures
            input_data = tx.get('input', '')
            if input_data and len(input_data) >= 10:
                function_sig = input_data[:10].lower()
                if function_sig in self.compound_functions:
                    is_compound = True
            
            # Check for cToken symbols in token transactions
            token_symbol = tx.get('tokenSymbol', '').lower()
            if token_symbol.startswith('c') and len(token_symbol) <= 6:
                is_compound = True
            
            if is_compound:
                # Add transaction type classification
                tx['compound_action'] = self._classify_transaction_type(tx)
                compound_related.append(tx)
        
        return compound_related
    
    def _classify_transaction_type(self, tx: Dict) -> str:
        """Classify the type of Compound transaction"""
        input_data = tx.get('input', '')
        
        if len(input_data) >= 10:
            function_sig = input_data[:10].lower()
            return self.compound_functions.get(function_sig, 'unknown')
        
        # Fallback classification based on value and context
        value = float(tx.get('value', 0))
        if value > 0:
            return 'supply'
        else:
            return 'unknown'

class EnhancedRiskScorer:
    """Enhanced risk scoring with more sophisticated algorithms"""
    
    def __init__(self):
        # Refined feature weights based on DeFi risk analysis
        self.feature_weights = {
            'leverage_risk': 0.30,      # Most important - how much leverage
            'liquidity_risk': 0.20,     # Market liquidity and position size
            'activity_risk': 0.15,      # Activity patterns
            'diversification_risk': 0.15, # Portfolio diversification
            'liquidation_risk': 0.10,   # Historical liquidations
            'behavioral_risk': 0.10     # Behavioral patterns
        }
        
        # Risk thresholds
        self.safe_ltv_ratio = 0.6      # 60% LTV considered safe
        self.high_risk_ltv = 0.8       # 80%+ LTV high risk
        self.min_healthy_txs = 50      # Minimum transactions for good scoring
        
    def calculate_wallet_metrics(self, transactions: List[Dict], wallet_address: str) -> WalletMetrics:
        """Enhanced metrics calculation"""
        metrics = WalletMetrics()
        
        if not transactions:
            # Return default metrics for wallets with no Compound activity
            metrics.frequency_score = 1.0  # High risk due to no activity
            return metrics
        
        # Parse and categorize transactions
        parsed_txs = self._enhanced_transaction_parsing(transactions)
        
        # Basic transaction metrics
        metrics.transaction_count = len(parsed_txs)
        
        # Calculate financial metrics
        supply_txs = [tx for tx in parsed_txs if tx.get('action') in ['mint', 'supply']]
        borrow_txs = [tx for tx in parsed_txs if tx.get('action') in ['borrow']]
        repay_txs = [tx for tx in parsed_txs if tx.get('action') in ['repayBorrow', 'repay']]
        liquidation_txs = [tx for tx in parsed_txs if tx.get('action') in ['liquidateBorrow', 'seize']]
        
        # Calculate totals (in USD equivalent, simplified)
        metrics.total_supplied = sum(tx.get('usd_value', 0) for tx in supply_txs)
        metrics.total_borrowed = sum(tx.get('usd_value', 0) for tx in borrow_txs)
        total_repaid = sum(tx.get('usd_value', 0) for tx in repay_txs)
        
        # Net position
        metrics.net_worth = metrics.total_supplied - metrics.total_borrowed + total_repaid
        
        # Liquidation count
        metrics.liquidation_count = len(liquidation_txs)
        
        # Token diversity
        unique_tokens = set()
        for tx in parsed_txs:
            if tx.get('token_symbol'):
                unique_tokens.add(tx['token_symbol'])
        metrics.unique_tokens = len(unique_tokens)
        
        # Calculate advanced metrics
        metrics.max_leverage_used = self._calculate_max_leverage(parsed_txs)
        metrics.position_concentration = self._calculate_concentration(parsed_txs)
        metrics.frequency_score = self._calculate_frequency_score(parsed_txs)
        metrics.gas_efficiency = self._calculate_gas_efficiency(parsed_txs)
        metrics.volatility_score = self._calculate_volatility(parsed_txs)
        
        # Time-based metrics
        if parsed_txs:
            timestamps = [tx.get('timestamp', 0) for tx in parsed_txs if tx.get('timestamp')]
            if timestamps:
                timestamps.sort()
                first_tx = min(timestamps)
                last_tx = max(timestamps)
                current_time = time.time()
                
                metrics.compound_usage_duration = (last_tx - first_tx) // (24 * 3600)
                
                # Recent activity (last 90 days)
                ninety_days_ago = current_time - (90 * 24 * 3600)
                recent_txs = sum(1 for ts in timestamps if ts > ninety_days_ago)
                metrics.recent_activity_ratio = recent_txs / len(timestamps) if timestamps else 0
                
                # Protocol loyalty (how long they've been active)
                days_since_last = (current_time - last_tx) // (24 * 3600)
                metrics.protocol_loyalty = max(0, 1 - (days_since_last / 365))  # Decay over a year
        
        return metrics
    
    def _enhanced_transaction_parsing(self, transactions: List[Dict]) -> List[Dict]:
        """Enhanced transaction parsing with USD value estimation"""
        parsed = []
        
        # Simple price mapping (in production, you'd use real price feeds)
        token_prices = {
            'ETH': 2000, 'WETH': 2000,
            'USDC': 1, 'USDT': 1, 'DAI': 1,
            'WBTC': 30000,
            'UNI': 6, 'LINK': 7, 'MKR': 1500,
            'SUSHI': 1, 'COMP': 50
        }
        
        for tx in transactions:
            parsed_tx = {
                'hash': tx.get('hash', ''),
                'timestamp': int(tx.get('timeStamp', 0)),
                'value': float(tx.get('value', 0)) / 1e18,
                'gas_used': int(tx.get('gasUsed', 0)),
                'gas_price': int(tx.get('gasPrice', 0)),
                'action': tx.get('compound_action', 'unknown'),
                'token_symbol': tx.get('tokenSymbol', 'ETH'),
                'to_address': tx.get('to', ''),
                'from_address': tx.get('from', ''),
            }
            
            # Estimate USD value
            token_symbol = parsed_tx['token_symbol'].replace('c', '').upper()
            price = token_prices.get(token_symbol, 1)
            parsed_tx['usd_value'] = parsed_tx['value'] * price
            
            # Calculate gas cost in USD
            gas_cost_eth = (parsed_tx['gas_used'] * parsed_tx['gas_price']) / 1e18
            parsed_tx['gas_cost_usd'] = gas_cost_eth * token_prices['ETH']
            
            parsed.append(parsed_tx)
        
        return parsed
    
    def _calculate_max_leverage(self, transactions: List[Dict]) -> float:
        """Calculate maximum leverage ratio ever used"""
        supply_total = 0
        borrow_total = 0
        max_leverage = 0
        
        for tx in sorted(transactions, key=lambda x: x.get('timestamp', 0)):
            action = tx.get('action', '')
            usd_value = tx.get('usd_value', 0)
            
            if action in ['mint', 'supply']:
                supply_total += usd_value
            elif action in ['borrow']:
                borrow_total += usd_value
            elif action in ['repayBorrow', 'repay']:
                borrow_total = max(0, borrow_total - usd_value)
            elif action in ['redeem', 'redeemUnderlying']:
                supply_total = max(0, supply_total - usd_value)
            
            # Calculate current leverage ratio
            if supply_total > 0:
                current_leverage = borrow_total / supply_total
                max_leverage = max(max_leverage, current_leverage)
        
        return max_leverage
    
    def _calculate_concentration(self, transactions: List[Dict]) -> float:
        """Calculate position concentration risk"""
        token_values = {}
        
        for tx in transactions:
            token = tx.get('token_symbol', 'UNKNOWN')
            value = tx.get('usd_value', 0)
            
            if token not in token_values:
                token_values[token] = 0
            token_values[token] += value
        
        if not token_values:
            return 1.0
        
        total_value = sum(token_values.values())
        if total_value == 0:
            return 1.0
        
        # Calculate Herfindahl index
        concentration = sum((value / total_value) ** 2 for value in token_values.values())
        return concentration
    
    def _calculate_frequency_score(self, transactions: List[Dict]) -> float:
        """Calculate transaction frequency score"""
        if not transactions:
            return 1.0  # High risk for no transactions
        
        timestamps = [tx.get('timestamp', 0) for tx in transactions if tx.get('timestamp')]
        if len(timestamps) < 2:
            return 0.8
        
        timestamps.sort()
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        
        if not intervals:
            return 0.8
        
        avg_interval_days = np.mean(intervals) / (24 * 3600)
        
        # Optimal frequency is around 7-30 days between transactions
        if 7 <= avg_interval_days <= 30:
            return 0.1  # Low risk
        elif avg_interval_days < 1:
            return 0.7  # High frequency might indicate automated/risky behavior
        else:
            return min(0.9, avg_interval_days / 100)  # Infrequent activity
    
    def _calculate_gas_efficiency(self, transactions: List[Dict]) -> float:
        """Calculate gas efficiency score"""
        if not transactions:
            return 0.5
        
        gas_costs = [tx.get('gas_cost_usd', 0) for tx in transactions if tx.get('gas_cost_usd', 0) > 0]
        if not gas_costs:
            return 0.5
        
        avg_gas_cost = np.mean(gas_costs)
        
        # Efficient users spend reasonable gas
        if avg_gas_cost < 10:  # Very efficient
            return 0.1
        elif avg_gas_cost < 50:  # Reasonable
            return 0.3
        else:  # Inefficient, might indicate poor decision making
            return min(0.8, avg_gas_cost / 100)
    
    def _calculate_volatility(self, transactions: List[Dict]) -> float:
        """Calculate transaction size volatility"""
        values = [tx.get('usd_value', 0) for tx in transactions if tx.get('usd_value', 0) > 0]
        if len(values) < 2:
            return 0.5
        
        coefficient_of_variation = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        return min(1.0, coefficient_of_variation)
    
    def calculate_risk_score(self, metrics: WalletMetrics) -> Tuple[float, Dict[str, float]]:
        """Enhanced risk scoring algorithm"""
        risk_components = {}
        
        # 1. Leverage Risk (most important)
        if metrics.total_supplied > 0:
            current_ltv = metrics.total_borrowed / metrics.total_supplied
            leverage_risk = min(current_ltv / self.high_risk_ltv, 1.0)
            # Add max leverage component
            max_leverage_risk = min(metrics.max_leverage_used / self.high_risk_ltv, 1.0)
            risk_components['leverage_risk'] = (leverage_risk * 0.6 + max_leverage_risk * 0.4)
        else:
            risk_components['leverage_risk'] = 1.0 if metrics.total_borrowed > 0 else 0.3
        
        # 2. Liquidity Risk
        if metrics.transaction_count == 0:
            risk_components['liquidity_risk'] = 1.0
        else:
            # Risk decreases with more transactions (more data = better assessment)
            tx_risk = max(0, 1 - (metrics.transaction_count / self.min_healthy_txs))
            # Add concentration risk
            concentration_risk = metrics.position_concentration
            risk_components['liquidity_risk'] = (tx_risk * 0.6 + concentration_risk * 0.4)
        
        # 3. Activity Risk
        frequency_risk = metrics.frequency_score
        recent_activity_risk = 1 - metrics.recent_activity_ratio  # Less recent activity = higher risk
        risk_components['activity_risk'] = (frequency_risk * 0.7 + recent_activity_risk * 0.3)
        
        # 4. Diversification Risk
        if metrics.unique_tokens == 0:
            risk_components['diversification_risk'] = 1.0
        else:
            # Risk decreases with more token diversity
            diversity_risk = max(0, 1 - (metrics.unique_tokens / 8))  # 8+ tokens = well diversified
            gas_efficiency_risk = metrics.gas_efficiency
            risk_components['diversification_risk'] = (diversity_risk * 0.8 + gas_efficiency_risk * 0.2)
        
        # 5. Liquidation Risk
        if metrics.liquidation_count > 0:
            risk_components['liquidation_risk'] = min(metrics.liquidation_count / 3, 1.0)
        else:
            risk_components['liquidation_risk'] = 0.0
        
        # 6. Behavioral Risk
        volatility_risk = metrics.volatility_score
        loyalty_risk = 1 - metrics.protocol_loyalty
        risk_components['behavioral_risk'] = (volatility_risk * 0.6 + loyalty_risk * 0.4)
        
        # Calculate weighted overall risk score
        overall_risk = sum(
            risk_components[component] * self.feature_weights[component]
            for component in risk_components
        )
        
        # Apply non-linear scaling for better distribution
        # This helps spread scores more evenly across the 0-1000 range
        scaled_risk = 1 - np.exp(-3 * overall_risk)  # Exponential scaling
        risk_score = scaled_risk * 1000
        
        # Ensure minimum score for wallets with no activity
        if metrics.transaction_count == 0:
            risk_score = max(risk_score, 600)  # Minimum 600 for no activity
        
        return risk_score, risk_components

def main():
    """Enhanced main function with better error handling and statistics"""
    
    # Configuration - Load API key from environment variable
    ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
    
    if ETHERSCAN_API_KEY:
        logger.info("Using Etherscan API key from environment variable")
    else:
        logger.warning("No Etherscan API key found in environment variables. Using free tier with rate limits.")
    
    # Initialize components
    fetcher = EnhancedCompoundDataFetcher(ETHERSCAN_API_KEY)
    scorer = EnhancedRiskScorer()
    
    # Read wallet addresses
    try:
        # Try different file names
        file_names = ['Wallet id - Sheet1.csv', 'wallet_addresses.csv', 'wallets.csv']
        df_wallets = None
        
        for file_name in file_names:
            try:
                if os.path.exists(file_name):
                    df_wallets = pd.read_csv(file_name)
                    logger.info(f"Loaded wallet addresses from {file_name}")
                    break
            except:
                continue
        
        if df_wallets is None:
            logger.error("No wallet file found. Please ensure one of these files exists: " + ", ".join(file_names))
            return
        
        # Get wallet addresses from first column
        wallet_column = df_wallets.columns[0]
        wallet_addresses = df_wallets[wallet_column].dropna().astype(str).unique().tolist()
        
        # Clean addresses
        wallet_addresses = [addr.strip().lower() for addr in wallet_addresses if addr.strip().startswith('0x')]
        
        logger.info(f"Processing {len(wallet_addresses)} unique wallet addresses")
        
    except Exception as e:
        logger.error(f"Error reading wallet file: {e}")
        return
    
    results = []
    detailed_analysis = []
    processing_stats = {'success': 0, 'failed': 0, 'no_compound_activity': 0}
    
    # Process each wallet with progress tracking
    for i, wallet_address in enumerate(wallet_addresses, 1):
        logger.info(f"Processing wallet {i}/{len(wallet_addresses)}: {wallet_address}")
        
        try:
            # Fetch transaction data
            transactions = fetcher.fetch_wallet_transactions(wallet_address)
            
            # Calculate metrics
            metrics = scorer.calculate_wallet_metrics(transactions, wallet_address)
            
            # Calculate risk score
            risk_score, risk_components = scorer.calculate_risk_score(metrics)
            
            # Store results
            final_score = max(0, min(1000, int(round(risk_score))))  # Ensure 0-1000 range
            results.append({
                'wallet_id': wallet_address,
                'score': final_score
            })
            
            # Store detailed analysis
            detailed_analysis.append({
                'wallet_id': wallet_address,
                'risk_score': final_score,
                'transaction_count': metrics.transaction_count,
                'total_borrowed': round(metrics.total_borrowed, 2),
                'total_supplied': round(metrics.total_supplied, 2),
                'net_worth': round(metrics.net_worth, 2),
                'unique_tokens': metrics.unique_tokens,
                'liquidation_count': metrics.liquidation_count,
                'max_leverage': round(metrics.max_leverage_used, 3),
                'position_concentration': round(metrics.position_concentration, 3),
                'leverage_risk': round(risk_components['leverage_risk'], 3),
                'liquidity_risk': round(risk_components['liquidity_risk'], 3),
                'activity_risk': round(risk_components['activity_risk'], 3),
                'diversification_risk': round(risk_components['diversification_risk'], 3),
                'liquidation_risk': round(risk_components['liquidation_risk'], 3),
                'behavioral_risk': round(risk_components['behavioral_risk'], 3)
            })
            
            # Update statistics
            if metrics.transaction_count == 0:
                processing_stats['no_compound_activity'] += 1
            else:
                processing_stats['success'] += 1
            
            logger.info(f"Wallet {wallet_address}: Risk Score = {final_score} ({metrics.transaction_count} txs)")
            
        except Exception as e:
            logger.error(f"Error processing wallet {wallet_address}: {e}")
            processing_stats['failed'] += 1
            
            # Add with high risk score for failed processing
            results.append({
                'wallet_id': wallet_address,
                'score': 850  # High risk for unanalyzable wallets
            })
    
    # Save results and generate reports
    if results:
        # Main results CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv('wallet_risk_scores.csv', index=False)
        logger.info(f"Results saved to 'wallet_risk_scores.csv' with {len(results)} wallets")
        
        # Detailed analysis CSV
        if detailed_analysis:
            df_analysis = pd.DataFrame(detailed_analysis)
            df_analysis.to_csv('detailed_risk_analysis.csv', index=False)
            logger.info("Detailed analysis saved to 'detailed_risk_analysis.csv'")
        
        # Generate comprehensive statistics
        scores = [result['score'] for result in results]
        
        logger.info("="*50)
        logger.info("FINAL RISK SCORING RESULTS")
        logger.info("="*50)
        logger.info(f"Total wallets processed: {len(results)}")
        logger.info(f"Successful analysis: {processing_stats['success']}")
        logger.info(f"No Compound activity: {processing_stats['no_compound_activity']}")
        logger.info(f"Failed processing: {processing_stats['failed']}")
        logger.info("-"*30)
        logger.info(f"Risk Score Statistics:")
        logger.info(f"  Mean: {np.mean(scores):.1f}")
        logger.info(f"  Median: {np.median(scores):.1f}")
        logger.info(f"  Min: {min(scores)}")
        logger.info(f"  Max: {max(scores)}")
        logger.info(f"  Std Dev: {np.std(scores):.1f}")
        
        # Risk distribution
        low_risk = sum(1 for s in scores if s < 300)
        medium_risk = sum(1 for s in scores if 300 <= s < 700)
        high_risk = sum(1 for s in scores if s >= 700)
        
        logger.info("-"*30)
        logger.info(f"Risk Distribution:")
        logger.info(f"  Low Risk (0-299): {low_risk} wallets ({low_risk/len(scores)*100:.1f}%)")
        logger.info(f"  Medium Risk (300-699): {medium_risk} wallets ({medium_risk/len(scores)*100:.1f}%)")
        logger.info(f"  High Risk (700-1000): {high_risk} wallets ({high_risk/len(scores)*100:.1f}%)")
        logger.info("="*50)
        
        # Save summary report
        summary_report = {
            'total_wallets': len(results),
            'successful_analysis': processing_stats['success'],
            'no_compound_activity': processing_stats['no_compound_activity'],
            'failed_processing': processing_stats['failed'],
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'std_score': np.std(scores),
            'low_risk_count': low_risk,
            'medium_risk_count': medium_risk,
            'high_risk_count': high_risk
        }
        
        with open('risk_scoring_summary.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        logger.info("Summary report saved to 'risk_scoring_summary.json'")
    
    else:
        logger.error("No results generated")

if __name__ == "__main__":
    main()