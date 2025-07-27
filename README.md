# Compound Wallet Risk Scoring Methodology

## Overview
This document explains the comprehensive approach used to analyze wallet addresses and assign risk scores from 0-1000 for Compound protocol users.

## Data Collection Method

### 1. Multi-Source Data Fetching
- **Etherscan API Integration**: Fetches normal transactions, internal transactions, and ERC20 token transfers
- **Enhanced Rate Limiting**: Adaptive rate limiting based on API key availability (1s for free tier, 0.3s with API key)
- **Comprehensive Contract Coverage**: Monitors both Compound V2 and V3 contracts including:
  - Comptroller contracts
  - All cToken contracts (cDAI, cUSDC, cETH, cWBTC, etc.)
  - V3 specific contracts (cUSDCv3, cETHv3, etc.)

### 2. Transaction Classification
- **Function Signature Analysis**: Identifies specific Compound actions (mint, borrow, repay, liquidate, etc.)
- **Contract Address Filtering**: Direct interactions with known Compound contracts
- **Token Symbol Recognition**: Identifies cToken transfers and interactions

### 3. Data Quality Enhancements
- **Multiple Transaction Types**: Normal, internal, and token transfers
- **USD Value Estimation**: Converts all transactions to USD equivalent using price feeds
- **Gas Cost Analysis**: Tracks transaction efficiency and user behavior

## Feature Selection Rationale

### Primary Risk Components (Weighted)

#### 1. Leverage Risk (30% weight)
**Rationale**: Leverage is the primary risk factor in DeFi lending protocols.
- **Current LTV Ratio**: total_borrowed / total_supplied
- **Maximum Historical Leverage**: Highest leverage ratio ever achieved
- **Risk Threshold**: 60% LTV considered safe, 80%+ high risk

#### 2. Liquidity Risk (20% weight)
**Rationale**: Market liquidity affects position management and liquidation risk.
- **Transaction Count**: More transactions provide better risk assessment data
- **Position Concentration**: Herfindahl index of token distribution
- **Minimum Threshold**: 50+ transactions for reliable assessment

#### 3. Activity Risk (15% weight)
**Rationale**: Activity patterns reveal user sophistication and engagement.
- **Transaction Frequency**: Optimal range is 7-30 days between transactions
- **Recent Activity Ratio**: Activity in last 90 days vs. total history
- **Protocol Loyalty**: Time since last transaction (decay function)

#### 4. Diversification Risk (15% weight)
**Rationale**: Diversification reduces correlation risk and concentration exposure.
- **Token Diversity**: Number of unique tokens used (8+ considered well-diversified)
- **Gas Efficiency**: Average gas costs indicate decision-making quality

#### 5. Liquidation Risk (10% weight)
**Rationale**: Historical liquidations directly indicate risk management failures.
- **Liquidation Count**: Number of past liquidation events
- **Liquidation Frequency**: Rate of liquidations relative to total activity

#### 6. Behavioral Risk (10% weight)
**Rationale**: Behavioral patterns indicate risk tolerance and sophistication.
- **Transaction Size Volatility**: Coefficient of variation in transaction sizes
- **Protocol Loyalty**: Sustained engagement vs. sporadic usage

### Advanced Metrics

#### Position Concentration
```
Herfindahl Index = Σ(token_value_share²)
```
Measures concentration risk across different tokens.

#### Maximum Leverage Calculation
Tracks the highest leverage ratio achieved by simulating position changes over time:
```
Leverage = total_borrowed / total_supplied
```

#### Frequency Score
Optimal transaction frequency analysis:
- **Under-trading**: >30 days between transactions (higher risk)
- **Optimal**: 7-30 days between transactions (lower risk)
- **Over-trading**: <1 day between transactions (potentially automated/risky)

## Scoring Method

### 1. Component Scoring (0-1 scale)
Each risk component is normalized to a 0-1 scale where:
- **0** = Lowest risk
- **1** = Highest risk

### 2. Weighted Aggregation
```
Overall_Risk = Σ(Component_Risk × Component_Weight)
```

### 3. Non-Linear Scaling
Applied exponential scaling for better score distribution:
```
Scaled_Risk = 1 - exp(-3 × Overall_Risk)
Final_Score = Scaled_Risk × 1000
```

### 4. Special Cases
- **No Compound Activity**: Minimum score of 600 (high risk due to lack of data)
- **Failed Analysis**: Score of 850 (very high risk for unanalyzable wallets)
- **Score Bounds**: Enforced 0-1000 range

## Risk Indicators Justification

### High-Risk Indicators
1. **High Leverage Ratio** (>80% LTV): Primary liquidation risk
2. **Past Liquidations**: Direct evidence of poor risk management
3. **Position Concentration**: Single-token exposure increases correlation risk
4. **Irregular Activity**: Either too frequent (automated) or too sparse (abandoned)
5. **Poor Gas Efficiency**: Indicates suboptimal decision-making
6. **No Recent Activity**: Positions may be abandoned or poorly monitored

### Low-Risk Indicators
1. **Conservative Leverage** (<60% LTV): Safe borrowing practices
2. **Diversified Positions**: Multiple tokens reduce correlation risk
3. **Consistent Activity**: Regular position management
4. **No Liquidation History**: Good risk management track record
5. **Efficient Gas Usage**: Indicates sophisticated user behavior
6. **Long-term Engagement**: Sustained protocol usage

### Medium-Risk Indicators
1. **Moderate Leverage** (60-80% LTV): Acceptable but monitored risk
2. **Limited Diversification**: 2-4 tokens used
3. **Seasonal Activity**: Periodic but not consistent engagement
4. **Moderate Transaction Sizes**: Neither micro nor whale-sized positions

## Model Validation & Calibration

### Score Distribution Targets
- **Low Risk (0-299)**: Conservative users, well-diversified, low leverage
- **Medium Risk (300-699)**: Moderate risk-takers, some concentration or leverage
- **High Risk (700-1000)**: High leverage, poor diversification, liquidation history

### Quality Metrics Tracked
1. **Processing Success Rate**: Percentage of wallets successfully analyzed
2. **Data Coverage**: Percentage of wallets with sufficient transaction history
3. **Score Distribution**: Ensuring reasonable spread across risk categories


## Usage Instructions

1. **Setup**: Install required packages and optionally add Etherscan API key
2. **Input**: Place wallet addresses in CSV file (first column)
3. **Execution**: Run the script to generate risk scores
4. **Output**: 
   - `wallet_risk_scores.csv`: Main results (wallet_id, score)
   - `detailed_risk_analysis.csv`: Comprehensive metrics
   - `risk_scoring_summary.json`: Summary statistics
