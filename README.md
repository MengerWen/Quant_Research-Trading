# Quantitative Research - Summer 2025 Internship

A comprehensive quantitative research repository containing cryptocurrency trading factors, machine learning models, and backtesting frameworks developed during a summer internship focused on digital asset trading strategies.

## 🎯 Overview

This repository contains the results of extensive quantitative research in cryptocurrency markets, with particular focus on:
- **Factor Engineering**: Custom-designed factors for ETH trading
- **Machine Learning Models**: Advanced ML models for BTC price prediction
- **Backtesting Framework**: Comprehensive backtesting system for quantitative investing

## 📊 Key Achievements

### ✅ Successful Factors
- **Keltner Channel Factor** (`250627_keltner.ipynb`)
- **Smart Money Factor** (`250702_聪明钱.ipynb`) 
- **Time Center Deviation** (`250708_时间重心偏离.ipynb`)
- **Volume Convergence Factor (VCF)** (`250716_VCF.ipynb`)
- **Realized Semi-variance Jump Variation (RSRJV)** (`250801_RSRJV.ipynb`)

## 🏗️ Repository Structure

```
📦 25 Summer/
├── 📁 Factors/           # Factor development and testing
├── 📁 Models/            # Machine learning models
├── 📁 实习生工作评估指标/  # Performance evaluation metrics
├── 📁 论文等/            # Research papers and references
└── 📄 frame_guide.pdf   # Framework documentation
```

## 🔬 Factor Categories

### 1. Technical Indicators
- **RSI Variations**: Multiple RSI-based factors with different timeframes
- **MACD Derivatives**: Enhanced MACD factors with volatility adjustments
- **KDJ Oscillators**: Adaptive KDJ factors with volume confirmation
- **Bollinger Bands**: Multiple BB variations for different market conditions

### 2. Volume-Price Analysis
- **OBV (On-Balance Volume)**: Volume momentum indicators
- **VWAP**: Volume-weighted average price factors
- **Volume-Price Divergence**: Cross-sectional analysis factors
- **ATR-based**: Average True Range normalized factors

### 3. Advanced Mathematical Factors

#### Jump Variation Factors (`250731_跳跃/`)
- **Realized Jump Variation (RJV)**
- **Realized Semi-variance Jump Variation (RSRJV)**  ✅
- **Signed Jump Variation (SJ)** ✅
- **Jump Arrival Rate Analysis**

#### Alpha158 Enhancements
- **KUP1**: Enhanced upper shadow analysis
- **RSV**: Relative strength value improvements ✅
- **SUMD**: Sum of differences with ATR adjustment ✅

#### Research-Based Factors
- **RSRS**: Resistance Support Relative Strength (multiple variants)
- **Cross-sectional Factors**: Kurtosis, momentum, entropy-based
- **Fractal Market Factors**: Market microstructure analysis ✅

### 4. Institutional Research Implementations

#### 中金量化 (CICC)
- Order splitting algorithms for different market impact scenarios
- Multi-timeframe execution strategies (1m, 15m intervals)
- Portfolio impact analysis (1%, 5% position sizes)

#### 开源证券 (KAIYUAN Securities)
- **ERR Factors**: Error correction mechanisms
- **VCF Enhancements**: Volume convergence with multi-scale analysis
- **Time Center Deviation**: Fractal market improvements ✅

#### 国盛金工 (Guosheng Securities)
- **Herding Behavior**: Market sentiment correlation analysis
- **Trend Factors**: Liquidity impact and momentum detection
- **Smart Money Flow**: Enhanced capital flow detection

#### 招商证券 (China Merchants Securities)
- **Alligator Lines**: Bill Williams technical analysis implementation

## 🤖 Machine Learning Models

### ConvLSTM Architecture
- **Multi-factor Integration**: Combines 200+ engineered factors
- **Deep Learning Pipeline**: Convolutional LSTM for time series prediction
- **Factor Selection**: Automated feature importance ranking
- **Backtesting Integration**: Seamless model-to-strategy pipeline

### Model Features
- **16-core Parallel Processing**: Optimized for high-performance computing
- **Factor Screening**: Automated selection of optimal predictive features
- **Performance Metrics**: Comprehensive evaluation framework

## 🔄 Backtesting Framework

### Core Features
- **Multi-asset Support**: BTC, ETH, SOL backtesting capabilities
- **Transaction Costs**: Realistic fee structures and slippage modeling
- **Position Management**: Both single and dual-direction strategies
- **Performance Analytics**: Comprehensive return and risk metrics

### Framework Components
- **Data Pipeline**: High-frequency data processing (1m, 15m intervals)
- **Signal Generation**: Factor-to-signal conversion algorithms
- **Risk Management**: Position sizing and drawdown controls
- **Benchmarking**: Comparative performance analysis

## 📈 Factor Performance

### Correlation Analysis
- Comprehensive factor correlation matrices
- Cross-asset factor stability testing
- IC (Information Coefficient) analysis for predictive power

### Success Rate Tracking
- ✅ **Working Factors**: Documented successful implementations
- ❌ **Failed Experiments**: Learning from unsuccessful approaches
- 🔄 **Ongoing Research**: Continuous improvement initiatives

## 🛠️ Technical Stack

- **Python**: Primary development language
- **Jupyter Notebooks**: Interactive research and development
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Machine Learning**: TensorFlow/PyTorch for deep learning models
- **Visualization**: Advanced plotting for factor analysis

## 📚 Research Papers

The repository includes relevant academic papers:
- Jump variation mathematics research
- Market microstructure studies  
- Factor investing methodologies
- Cryptocurrency market analysis

## 🎓 Learning Outcomes

This internship project demonstrates:
- **Advanced Factor Engineering**: Created 100+ unique trading factors
- **Machine Learning Integration**: Built end-to-end ML trading pipeline
- **Research Implementation**: Translated academic research into practical factors
- **Performance Evaluation**: Comprehensive backtesting and validation

## 📋 Usage

1. **Factor Development**: Navigate to `Factors/` for individual factor implementations
2. **Model Training**: Use `Models/` for machine learning model development
3. **Backtesting**: Leverage the framework for strategy validation
4. **Research**: Reference `论文等/` for theoretical foundations

## 🔮 Future Work

- **Real-time Implementation**: Deploy successful factors in live trading
- **Multi-asset Expansion**: Extend to additional cryptocurrency pairs
- **Risk Model Enhancement**: Advanced portfolio risk management
- **Alternative Data Integration**: Incorporate sentiment and on-chain metrics

---

**Note**: This repository represents comprehensive quantitative research conducted during a summer 2025 internship, focusing on cryptocurrency market factor development and machine learning applications in algorithmic trading.
