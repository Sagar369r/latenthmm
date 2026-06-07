import os
import sys
import json
import time
import polars as pl
from datetime import datetime

# Twisted for async loop
from twisted.internet import reactor, task

# cTrader Open API
from ctrader_open_api import Client, EndPoints, TcpProtocol
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *

# Load V4 Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v4_engine.live_bridge import LiveExecutionEngine

# --- CONFIGURATION (User Must Set These) ---
APP_CLIENT_ID = os.getenv("CTRADER_CLIENT_ID", "30182_MEqHsNhzTMosn67W6z0tsBgdFGDciieiP59eKzoqCGlO1YOYJY")
APP_CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET", "jOGmayLGU1y4GFs8i2kti9jQLnzqdVLTArQdnVS1oIxFSQFqX0")
ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID", "5835062")
ACCOUNT_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")

# Target Assets
TARGET_SYMBOLS = ["AUDCAD", "CHFJPY", "EURNZD"]
# --------------------------------------------

class CTraderBridge:
    def __init__(self):
        print("🚀 Initializing Latent-Diffusion cTrader Bridge...")
        self.engine = LiveExecutionEngine()
        self.client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.client.setConnectedCallback(self.on_connected)
        self.client.setDisconnectedCallback(self.on_disconnected)
        self.client.setMessageReceivedCallback(self.on_message)
        
        self.symbols_map = {} # Map name to symbol_id
        self.reverse_symbols_map = {} # Map symbol_id to name
        self.historical_data = {s: [] for s in TARGET_SYMBOLS}
        
    def start(self):
        print("🔗 Connecting to cTrader Open API...")
        # Reactor runs asynchronously
        reactor.run()
        
    def on_connected(self):
        print("✅ Connected! Authenticating Application...")
        req = ProtoOAApplicationAuthReq()
        req.clientId = APP_CLIENT_ID
        req.clientSecret = APP_CLIENT_SECRET
        self.client.send(req)
        
    def on_disconnected(self, reason):
        print(f"❌ Disconnected: {reason}")
        reactor.stop()
        
    def on_message(self, message):
        payloadType = message.payloadType
        
        # Application Auth Response
        if payloadType == ProtoOAApplicationAuthRes().payloadType:
            print("✅ Application Authenticated. Authorizing Account...")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = int(ACCOUNT_ID)
            req.accessToken = ACCOUNT_TOKEN
            self.client.send(req)
            
        # Account Auth Response
        elif payloadType == ProtoOAAccountAuthRes().payloadType:
            print(f"✅ Account {ACCOUNT_ID} Authenticated. Fetching Symbol List...")
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = int(ACCOUNT_ID)
            self.client.send(req)
            
        # Symbol List Response
        elif payloadType == ProtoOASymbolsListRes().payloadType:
            res = ProtoOASymbolsListRes()
            res.ParseFromString(message.payload)
            
            for sym in res.symbol:
                if sym.symbolName in TARGET_SYMBOLS:
                    self.symbols_map[sym.symbolName] = sym.symbolId
                    self.reverse_symbols_map[sym.symbolId] = sym.symbolName
                    print(f"  🔍 Found {sym.symbolName} (ID: {sym.symbolId})")
            
            print("📊 Requesting historical 1-Hour Trendbars for AI Context...")
            self.fetch_historical_bars()
            
        # Historical Data Response
        elif payloadType == ProtoOAGetTrendbarsRes().payloadType:
            res = ProtoOAGetTrendbarsRes()
            res.ParseFromString(message.payload)
            
            symbol_name = self.reverse_symbols_map.get(res.symbolId, "Unknown")
            print(f"📥 Received {len(res.trendbar)} historical bars for {symbol_name}")
            
            # Format and process through ONNX Engine
            self.process_symbol_inference(symbol_name, res.trendbar)
            
        # Order Execution Response
        elif payloadType == ProtoOAExecutionEvent().payloadType:
            res = ProtoOAExecutionEvent()
            res.ParseFromString(message.payload)
            print(f"⚡ Execution Event: Order {res.order.orderId} status: {res.executionType}")
            
        # Error Message
        elif payloadType == ProtoOAErrorRes().payloadType:
            res = ProtoOAErrorRes()
            res.ParseFromString(message.payload)
            print(f"❌ API Error: {res.errorCode} - {res.description}")

    def fetch_historical_bars(self):
        # We need the last 150 hours to feed the Diffusion Engine
        now = int(time.time() * 1000)
        start_time = now - (150 * 60 * 60 * 1000) 
        
        for symbol, sym_id in self.symbols_map.items():
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = int(ACCOUNT_ID)
            req.period = ProtoOATrendbarPeriod.H1
            req.symbolId = sym_id
            req.fromTimestamp = start_time
            req.toTimestamp = now
            self.client.send(req)
            
    def process_symbol_inference(self, symbol, trendbars):
        # Convert Protobuf Trendbars to List of Dicts for Polars
        candles = []
        for bar in trendbars:
            candles.append({
                "time": bar.utcTimestampInMinutes * 60,
                "open": bar.low + bar.deltaOpen,
                "high": bar.low + bar.deltaHigh,
                "low": bar.low,
                "close": bar.low + bar.deltaClose,
                "volume": bar.volume
            })
            
        # Scale cTrader Prices (cTrader sends prices as integers scaled by 10^5)
        df = pl.DataFrame(candles)
        df = df.with_columns([
            (pl.col("open") / 100000.0),
            (pl.col("high") / 100000.0),
            (pl.col("low") / 100000.0),
            (pl.col("close") / 100000.0)
        ])
        
        print(f"\n🧠 Running V4 Inference for {symbol}...")
        decision = self.engine.process_live_tick(df)
        
        if decision['action'] == 'EXECUTE':
            print(f"🚨 {symbol} VETO PASSED! Absolute Confidence: {decision['p_final']:.4f}")
            self.execute_order(symbol, decision)
        else:
            print(f"🛡️ {symbol} Veto Active. Trade Killed.")

    def execute_order(self, symbol, decision):
        # Calculate Lots (Assuming 0.01 micro lots for testing)
        volume = 1000 
        
        direction = ProtoOATradeSide.BUY if decision['direction'] == 'BUY' else ProtoOATradeSide.SELL
        
        # cTrader Open API expects relative SL/TP as an integer in 1/100,000 of price
        sl_pips = int(decision['sl_distance'] * 100000)
        tp_pips = int(decision['tp_distance'] * 100000)
        
        print(f"🚀 Firing MARKET {decision['direction']} Order for {symbol} - Vol: {volume} | SL: {sl_pips} | TP: {tp_pips}")
        
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = int(ACCOUNT_ID)
        req.symbolId = self.symbols_map[symbol]
        req.orderType = ProtoOAOrderType.MARKET
        req.tradeSide = direction
        req.volume = volume
        req.relativeStopLoss = sl_pips
        req.relativeTakeProfit = tp_pips
        req.comment = f"V4_Sniper_P={decision['p_final']:.2f}"
        
        self.client.send(req)

if __name__ == '__main__':
    if APP_CLIENT_ID == "YOUR_CLIENT_ID_HERE":
        print("⚠️ ERROR: You must set your cTrader APP_CLIENT_ID in the script!")
        sys.exit(1)
        
    bridge = CTraderBridge()
    bridge.start()
