"""
股票服務模組
提供台股與美股相關數據的查詢功能
"""

import os  # 匯入作業系統模組，可操作環境變數、路徑、檔案系統（如 os.getenv、os.path.join 等）
import logging  # 匯入日誌模組，用於輸出系統訊息（info、debug、error 等級），方便除錯與追蹤執行流程
import yfinance as yf  # ✅ 匯入 Yahoo Finance 套件並簡寫為 yf，可用來抓取股市資料（歷史價格、即時數據等）
from typing import Tuple

# 👉 以下為 llama-index（前稱 GPT Index）相關匯入，支援 Function Calling Agent 架構
from llama_index.core.agent import FunctionCallingAgentWorker  # ✅ 建立具備 Function Calling 能力的 agent 工作者，可呼叫定義好的 tool 函式
from llama_index.core.tools import FunctionTool  # ✅ 將 Python 函式封裝成可被 LLM 呼叫的工具（Tool），供 Agent 使用
from llama_index.llms.openai import OpenAI  # ✅ 匯入 OpenAI 作為底層語言模型的驅動器（LLM Provider）


# 🔍 FunctionCallingAgent 的工具調用補充說明：
# ---------------------------------------------------------------
# 當我們使用 worker.as_agent().chat(prompt) 發送請求時，
# LLM 會根據提示內容（prompt）自行判斷是否需要呼叫任何已註冊的 FunctionTool。
# 
# ✅ 若 LLM 認為自己能夠直接回答（即便有 tool），它可能「完全不使用工具」。
# ✅ 若工具都不適用或 prompt 與工具無關（例如查詢非股票資訊），
#    LLM 會忽略所有工具，直接以自身知識生成回答，不會報錯或中斷。
# 
# ⚠️ 這代表當工具未被觸發時，回傳的內容仍是 LLM 生成的結果，但可能：
#    - 沒有使用即時資料（如股價、新聞）
#    - 未遵循資料一致性（因為沒走我們的工具）

# ⚠️ 注意：LLM 具備強大內部知識，若覺得問題能自行回答，可能會忽略我們註冊的工具（FunctionTool）
#    → 特別是當工具輸出太簡略或 prompt 沒明確引導時更容易發生
# ✅ 建議強化工具輸出品質，並在提示中明確要求「務必使用工具」，以提升工具被觸發的可能性
# ---------------------------------------------------------------


# 設定 logging 的輸出等級為 INFO：表示只輸出 info 級別以上（包含 warning、error）的訊息:
# ✅ 此行設定 logging 的最低顯示等級為 INFO（含以上級別），
#    因此僅會輸出下列等級的日誌訊息：
#       - logging.info(...)
#       - logging.warning(...)
#       - logging.error(...)
#       - logging.critical(...)
# ⚠️ logging.debug(...) 不會被顯示（因為等級低於 INFO，會被過濾掉）
# 
# 🛠 若需要顯示 debug 訊息，請改成：
#     logging.basicConfig(level=logging.DEBUG)
# 
# 🔁 注意：basicConfig 只能設定一次，多次呼叫不會生效，
#        若需動態調整等級，請改用 logging.getLogger() 的方式。
logging.basicConfig(level=logging.INFO)




# ════════════════════════════════════════════════════════════════════════
# 🧠 StockService 類別
# ════════════════════════════════════════════════════════════════════════
# 本類別封裝股票分析邏輯，結合 yfinance 與 llama-index 的 Agent 系統，
# 適用於：
#    🔹 查詢股票價格、公司資訊、財務數據、分析師建議、新聞等
#    🔹 提供可被 LLM 自動調用的工具（FunctionTool）
#    🔹 作為智能分析 Agent（GPT-4o 等）中的股票分析模組
#
# 核心功能包含：
# ✅ 整合 yfinance API：統一查詢股票數據
# ✅ 註冊 FunctionTool：每個功能對應一個 callable 工具
# ✅ 搭建 LLM Agent：支援 Function Calling，自動調度工具完成任務
# ✅ 智能分析股票：依據一組提示自動綜合股價/財務/新聞等資訊提出建議
#
# 常用場景：可作為 RAG 查詢組件、Chainlit 智能問答分析器、金融分析服務模組等
#
# ════════════════════════════════════════════════════════════════════════
# 🔧 初始化元件：
# - self.api_key           ► OpenAI 金鑰，供 Agent 模型使用
# - self.agent             ► 使用 llama-index 建立的 FunctionCallingAgent
#
# 🧩 工具列表（FunctionTool）註冊如下：
# - get_stock_price             ► 查詢當前股票收盤價（收盤點位）
# - get_company_info            ► 擷取公司全名、產業分類與簡介摘要
# - get_financial_ratios        ► 提取 P/E、P/B、殖利率等關鍵財務指標
# - get_analyst_recommendations ► 最新分析師評級與時間
# - get_recent_news             ► 查詢最新一則重大新聞（標題與連結）
# - get_industry_comparison     ► 比較公司 P/E 與產業平均，判斷相對高估或低估
#
# ════════════════════════════════════════════════════════════════════════
# 🧱 結構與方法總覽：
#
# 1. __init__(self, openai_api_key: Optional[str] = None)
#    ✅ 初始化：設定 API 金鑰並建立 Agent
#
# 2. _initialize_agent(self)
#    ✅ 註冊工具、建立 OpenAI LLM 並綁定 FunctionCallingAgent
#
# 3. get_stock_price(self, symbol: str) -> str
#    ✅ 抓取即時收盤價（當日 close）
#
# 4. get_company_info(self, symbol: str) -> str
#    ✅ 擷取公司名稱、產業別與業務摘要
#
# 5. get_financial_ratios(self, symbol: str) -> str
#    ✅ 查詢 P/E、P/B、Dividend Yield 並格式化輸出
#
# 6. get_analyst_recommendations(self, symbol: str) -> str
#    ✅ 回傳最新一筆分析師建議與評級時間
#
# 7. get_recent_news(self, symbol: str) -> str
#    ✅ 查詢最近一則新聞標題與連結
#
# 8. get_industry_comparison(self, symbol: str) -> str
#    ✅ 比較該股本益比與同產業平均，推論估值狀態
#
#    [主要執行程式]
# 9. analyze_stock(self, stock_symbol: str) -> Tuple[bool, str]
#    ✅ 綜合使用所有工具，由 Agent 自動分析是否建議買入/持有/賣出
#
# ════════════════════════════════════════════════════════════════════════

class StockService:
    """
    股票服務類別
    提供整合多種股票分析工具的服務介面，支援查詢價格、公司資訊、
    財務比率、分析師建議、新聞、以及行業比較等功能，並結合 LLM 進行智慧分析。
    """
    
    def __init__(self, openai_api_key=None):
        """
        初始化股票服務類別
        - 讀取 OpenAI API 金鑰（可由參數或環境變數取得）
        - 初始化內部 agent 工具
        """
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")  # 優先使用傳入金鑰，否則讀取環境變數
        self.agent = None  # 預設 agent 為 None，稍後由 def _initialize_agent() 建立
        self._initialize_agent() # 呼叫內部方法(下一個def)，完成 agent 初始化與 tools 註冊


    def _initialize_agent(self):
        """
        初始化 Function Calling Agent 與相關股票分析工具
        - 將內建函式封裝成 FunctionTool
        - 使用 OpenAI GPT-4o-mini 作為語言模型
        - 建立 FunctionCallingAgentWorker 與專業 prompt
        """

        # 建立 agent 可使用的工具函式清單（全部以 FunctionTool 形式封裝）:
        # 下述tools中每個def 都在後續的 def 中 

        # 📌 FunctionTool 與參數推斷補充說明：
        #     FunctionTool.from_defaults(fn=...) 會自動根據目標函式的簽名與型別註解（如 symbol: str）
        #     解析出參數結構（JSON Schema），供 LLM 理解該工具的用途與輸入格式，無需手動標明輸入的參數。
        #     ⭐EX: fn=self.get_stock_price  而不是  fn=self.get_stock_price(self, symbol: str)
        tools = [
            FunctionTool.from_defaults(fn=self.get_stock_price),   # ✅ 查詢股票當前收盤價（Close），用於評估市場即時定價
            FunctionTool.from_defaults(fn=self.get_company_info),  # ✅ 擷取公司全名、所屬產業與簡介摘要，協助判斷公司基本面
            FunctionTool.from_defaults(fn=self.get_financial_ratios), # ✅ 取得 P/E、P/B、股息殖利率等關鍵財務指標，衡量估值與報酬
            FunctionTool.from_defaults(fn=self.get_analyst_recommendations),  # ✅ 抽取最新分析師評級與日期，參考市場專業預期趨勢
            FunctionTool.from_defaults(fn=self.get_recent_news),  #  ✅ 擷取最新一則新聞標題與連結，掌握可能影響股價的重大事件
            FunctionTool.from_defaults(fn=self.get_industry_comparison)  # ✅ 將該股 P/E 與產業平均值比較，評估相對高估或低估狀況
        ]


        # 初始化語言模型：使用 OpenAI 的 GPT-4o-mini，並指定溫度為 0（穩定、可控）
        llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)

        # 創建 FunctionCallingAgentWorker 並綁定工具與模型，並設定任務導向的 system_prompt
        worker = FunctionCallingAgentWorker.from_tools(
            tools,   # 提供 agent 可用的所有工具函式
            llm=llm,  # 使用的 LLM 模型
            system_prompt="""你是一位專業的股票分析師。你的任務是分析給定的股票，並根據所有可用資訊提供是否購買的建議。 
            請使用所有可用工具來收集相關資訊，然後給出全面的分析和明確的建議。 
            考慮當前價格、公司資訊、財務比率、分析師推薦、最新新聞以及行業比較。 
            解釋你的推薦理由，並提供一個清晰的「買入」、「持有」或「賣出」建議。 
            如果某些資訊無法取得，請在分析中說明，並基於可用資訊給出最佳判斷。
            """
        )

        # ✅ 將建立好的 worker 轉換為通用 agent，之後可直接呼叫 self.agent.chat(prompt) 使用 => 最後一個 def analyze_stock() 有使用到!!
        self.agent = worker.as_agent()
    


    def get_stock_price(self, symbol: str) -> str:
        """
        🔹 查詢指定股票代碼（symbol）的當前收盤價（即最新的 Close 價格）
        🔹 使用 yfinance 套件抓取當日的歷史價格資料
        🔹 若成功取得資料，回傳一段格式化的價格字串
        🔹 若資料為空或發生例外錯誤，則回傳提示訊息
        Args:
            symbol (str): 股票代碼（如 'AAPL', 'TSLA'）
        Returns:
            str: 包含價格資訊或錯誤訊息的字串
        """
        try:
            # 建立 yfinance 的 Ticker 物件，會根據傳入的股票代碼 symbol 對應 Yahoo Finance 上的資料來源
            # stock 物件包含如 .info（公司資料）、.history（歷史股價）、.actions（分割/配息）等豐富屬性
            stock = yf.Ticker(symbol) 
            # 透過 .history() 方法抓取最近 1 天的歷史股價（回傳值為 pandas 的 DataFrame）
            # 這個 DataFrame 包含：Open、High、Low、Close、Volume、Dividends、Stock Splits 等欄位
            data = stock.history(period="1d") 

            # 確認資料不為空（data.empty 為 False 代表成功抓到當日資料）:
            # data.empty 是 pandas 的內建屬性，用來判斷整個 DataFrame 是否為空
            # ✅不是取名為 'empty' 的欄位 → 而是檢查是否有資料（rows == 0）
            if not data.empty:
                #  從 DataFrame 中取出「Close」（收盤價）欄位的最後一筆值（iloc[-1]）
                #  因為即使是 1 天的資料，也可能有多筆時間點，這裡取最後的收盤價
                current_price = data['Close'].iloc[-1]
                return f"The current price of {symbol} is ${current_price:.2f}"  # 格式化價格為浮點數小數點後兩位（例如 $173.45）
            
            # 若 data 為空，表示沒有成功抓到任何當日資料（可能是代碼錯誤、非交易日等原因）
            else:
                return f"Unable to fetch the current price for {symbol}. The stock data is empty."
            
        # ⚠️ 若在資料抓取過程中發生任何例外（如網路錯誤、symbol 無效、API 限制等）
        #    則記錄錯誤訊息到 logging 系統（日誌）以利除錯追蹤
        except Exception as e:
            logging.error(f"Error fetching stock price for {symbol}: {str(e)}")
            # 同時回傳錯誤訊息給前端/使用者（避免整個程式崩潰）
            return f"Error fetching stock price for {symbol}: {str(e)}"  



    def get_company_info(self, symbol: str) -> str:
        """
        🔹 根據輸入的股票代碼（symbol）查詢該公司的基本資訊
        🔹 包括：公司完整名稱（longName）、產業別（sector）、公司業務摘要（長敘述簡介，截斷前 200 字）
        🔹 若取得成功，回傳格式化的描述字串；若抓取失敗或資料不完整，則顯示錯誤訊息或保底值
        Args:
            symbol (str): 股票代碼，如 'AAPL', 'GOOG', 'TSMC'
        Returns:
            str: 描述公司基本資訊的字串，或錯誤提示
        """
        try:
            stock = yf.Ticker(symbol) # 建立 yfinance 的 Ticker 物件，可用於查詢股票相關資訊
            # ✅ 使用 .info 屬性抓取該股票對應的公司資訊
            #    注意：這會回傳一個 dict，內容包含公司名、行業、員工數、地址、市值、簡介等
            #    例：info['longName'], info['sector'], info['longBusinessSummary'] ...等欄位
            info = stock.info

            # ✅ 回傳格式化資訊：
            #    - 公司完整名稱（如：Apple Inc.）=> info['longName']
            #    - 股票代碼 => symbol
            #    - 所屬產業（若無此欄位則使用 'Unknown' 作為保底）=> info['sector']
            #    - 公司長敘述的前 200 個字（可能為英文簡介）=> info['longBusinessSummary']
            return f"{info['longName']} ({symbol}) is in the {info.get('sector', 'Unknown')} sector. {info.get('longBusinessSummary', '')[:200]}..."

        # 若抓取過程中出現錯誤（如 symbol 錯誤、無法連線、info 欄位缺失等），記錄錯誤 log
        except Exception as e:
            logging.error(f"Error fetching company info for {symbol}: {str(e)}")
            # 回傳錯誤訊息，確保即使出錯也能回應前端/使用者，而不是整個中斷
            return f"Error fetching company info for {symbol}: {str(e)}" 



    def get_financial_ratios(self, symbol: str) -> str:
        """
        🔹 查詢指定股票代碼（symbol）對應的財務比率指標
        🔹 包括：P/E（本益比）、P/B（股價淨值比）、Dividend Yield（股息殖利率）
        🔹 所有資料均來自 yfinance 提供的 info 字典，若無資料則顯示 N/A
        Args:
            symbol (str): 股票代碼，如 'AAPL', 'GOOG', 'TSMC'
        Returns:
            str: 格式化後的財務比率資訊，或錯誤訊息
        """
        try:
            stock = yf.Ticker(symbol)  # 建立 yfinance 的 Ticker 物件，用來查詢該股票的資訊
            info = stock.info  # 透過 .info 屬性獲取完整的公司基本資料字典（dict）

            # ✅ 嘗試從 info 中取得財務指標：
            #    trailingPE → 本益比（Price-to-Earnings Ratio）：股價除以每股盈餘
            #    priceToBook → 股價淨值比（Price-to-Book Ratio）：股價除以帳面淨值
            #    dividendYield → 股息殖利率（以比例表示，如 0.015 表示 1.5%）
            pe_ratio = info.get('trailingPE', 'N/A')  # 若無此欄位，則顯示為 'N/A'
            pb_ratio = info.get('priceToBook', 'N/A') # 若無此欄位，則顯示為 'N/A'
            dividend_yield = info.get('dividendYield', 'N/A') # 若無此欄位，則顯示為 'N/A'

            # 若成功取得 dividend_yield，則將其轉為百分比格式（浮點 → %）
            if dividend_yield != 'N/A':
                dividend_yield = f"{dividend_yield * 100:.2f}%"

            # 回傳格式化後的比率資訊：標明代碼與各項比率    
            return f"{symbol} financial ratios: P/E: {pe_ratio}, P/B: {pb_ratio}, Dividend Yield: {dividend_yield}"
        

        # 若抓取或計算過程中出現任何錯誤（如資料缺失、除法錯誤、連線問題）
        # 記錄錯誤到 log，以利後續除錯
        except Exception as e:
            logging.error(f"Error fetching financial ratios for {symbol}: {str(e)}")
            # 同時回傳錯誤訊息給使用端（保證函式永不中斷）
            return f"Error fetching financial ratios for {symbol}: {str(e)}"



    def get_analyst_recommendations(self, symbol: str) -> str:
        """
        🔹 根據輸入的股票代碼（symbol），查詢最新一筆分析師對該股票的推薦等級
        🔹 推薦等級可能包括「Buy」、「Hold」、「Sell」等變化
        🔹 資料來源為 yfinance 套件中的 recommendations 屬性（DataFrame 格式）
        Args:
            symbol (str): 股票代碼（如 'TSLA', 'AAPL', 'AMD'）
        Returns:
            str: 最新分析師建議與評等日期，或提示無資料 / 錯誤訊息
        """
        try:
            stock = yf.Ticker(symbol) # 建立 yfinance 股票物件，供後續查詢各類分析數據

            # ✅ 抓取分析師推薦資料（DataFrame），可能包含歷史多筆記錄
            #    recommendations.index 通常是 datetime 格式（表示每筆建議的日期）
            #    每筆紀錄欄位可能有：Firm、To Grade、From Grade、Action 等
            recommendations = stock.recommendations

            # 判斷是否成功取得推薦資料，且內容非空
            if recommendations is not None and not recommendations.empty:
                latest_rec = recommendations.iloc[-1] # 取出最新一筆推薦資料（DataFrame 的最後一行）=> 可看 D:\Python筆記\Pandas筆記\df 取用、更改(行列)_範例4
                # 回傳推薦評等（To Grade）與其對應日期（用 .name.date() 取出索引中的日期部分）
                return f"Latest analyst recommendation for {symbol}: {latest_rec['To Grade']} as of {latest_rec.name.date()}"
            
            # 若 recommendations 為 None 或為空 DataFrame，表示無可用的分析師建議
            else:
                return f"No analyst recommendations available for {symbol}"
        
        # ⚠️ 若資料抓取或處理過程發生例外（如無法連線、欄位缺失、symbol 錯誤等）
        #    記錄錯誤細節到 log 檔以利除錯
        except Exception as e:
            logging.error(f"Error fetching analyst recommendations for {symbol}: {str(e)}")
            # 同時回傳錯誤訊息給使用端（保證函式永不中斷）
            return f"Unable to fetch analyst recommendations for {symbol} due to an error: {str(e)}"



    def get_recent_news(self, symbol: str) -> str:
        """
        🔹 根據輸入的股票代碼（symbol），查詢該公司最新一則新聞
        🔹 使用 yfinance 的 Ticker.news 屬性（回傳為列表，每筆為新聞字典）
        🔹 若成功取得，回傳新聞標題與超連結；若無新聞或發生錯誤，則給出提示訊息
        Args:
            symbol (str): 股票代碼，例如 'AAPL', 'TSLA', 'GOOG'
        Returns:
            str: 一則格式化的新聞標題與連結，或錯誤/無資料訊息
        """
        try:
            stock = yf.Ticker(symbol)  # 建立 yfinance 股票物件，準備查詢該股票的相關資料

            # ✅ 抓取該股票的最新新聞（回傳為 list of dict，每個 dict 代表一篇新聞）
            #    每篇新聞通常包含 title（標題）、link（連結）、publisher、providerPublishTime 等欄位
            news = stock.news

            # 判斷是否成功取得新聞資料（即：news 為非空 list）
            if news:
                latest_news = news[0] # 取出最新的一則新聞（第一筆資料，通常為最新時間）
                # 組合標題與連結，並回傳給使用者
                return f"Latest news for {symbol}: {latest_news['title']} - {latest_news['link']}"
            
            # 若 news 為空 list，表示目前沒有可用的新聞資料
            else:
                return f"No recent news available for {symbol}"
            
        # ⚠️ 若在資料抓取或處理過程中發生任何例外（如 API 錯誤、symbol 錯誤、格式錯誤）
        #    將錯誤細節寫入 log 中供後續除錯
        except Exception as e:
            logging.error(f"Error fetching recent news for {symbol}: {str(e)}")
            # 同時回傳錯誤訊息給使用端（保證函式永不中斷）
            return f"Error fetching recent news for {symbol}: {str(e)}"



    def get_industry_comparison(self, symbol: str) -> str:
        """
        🔹 查詢指定股票（symbol）所屬產業與行業，並比較其本益比（P/E）與該行業的平均值
        🔹 分析該股票是否可能被低估或高估
        Args:
            symbol (str): 股票代碼，例如 'AAPL', 'TSLA'
        Returns:
            str: 格式化的行業比較分析結果，或錯誤訊息
        """
        try:
            stock = yf.Ticker(symbol)  # 建立 yfinance 的股票物件
            info = stock.info  # 抓取該股票的公司資訊（以 dict 格式提供）

            # ✅ 提取所屬產業（sector）與細分類別（industry）
            #    如果欄位缺失則顯示 'Unknown'
            sector = info.get('sector', 'Unknown') 
            industry = info.get('industry', 'Unknown')

            # ✅ 提取該股票本身的 P/E（本益比）與所在行業平均 P/E
            #    trailingPE: 公司近 12 個月的本益比
            #    industryPE: 同一行業的平均本益比（若 yfinance 有提供）
            pe_ratio = info.get('trailingPE', 'N/A')  # 該股票的本益比
            industry_pe = info.get('industryPE', 'N/A')  # 該產業平均本益比

            # 組合基本描述句，先說明所屬產業與行業
            comparison = f"{symbol} is in the {sector} sector, specifically in the {industry} industry. \n"

            # 若 P/E 與行業平均 P/E 都有資料，則進行比較分析
            if pe_ratio != 'N/A' and industry_pe != 'N/A':
                if pe_ratio < industry_pe:  # ✅ P/E 低於行業 → 可能被低估（價值被市場忽略）
                    comparison += f"Its P/E ratio ({pe_ratio:.2f}) is lower than the industry average ({industry_pe:.2f}), which could indicate it's undervalued compared to its peers."
                elif pe_ratio > industry_pe:  # ✅ P/E 高於行業 → 可能被高估（投資人期待過高）
                    comparison += f"Its P/E ratio ({pe_ratio:.2f}) is higher than the industry average ({industry_pe:.2f}), which could indicate it's overvalued compared to its peers."
                else:  # ✅ 與平均一致 → 評價中性
                    comparison += f"Its P/E ratio ({pe_ratio:.2f}) is in line with the industry average ({industry_pe:.2f})."
            
            #  ⚠️ 若任一數據缺失，無法進行有效比較
            else:
                comparison += "Unable to compare P/E ratio with industry average due to lack of data."

            # ✅ 回傳組合好的分析結論
            return comparison
        

        # ⚠️ 若在資料抓取或處理過程中發生任何例外（如 API 錯誤、symbol 錯誤、格式錯誤）
        #    將錯誤細節寫入 log 中供後續除錯
        except Exception as e:
            logging.error(f"Error fetching industry comparison for {symbol}: {str(e)}")
            # 同時回傳錯誤訊息給使用端（保證函式永不中斷）
            return f"Unable to fetch industry comparison for {symbol} due to an error: {str(e)}"


    # [主要執行程式]    
    def analyze_stock(self, stock_symbol: str) -> Tuple[bool, str]:
        """
        🔹 綜合分析指定股票的投資價值（是否值得買進、持有或賣出）
        🔹 使用 FunctionCallingAgent 與事先註冊的工具函式進行多面向分析
        🔹 自動處理(台、美)股代碼、格式轉換，並構建清晰的自然語言提示給 Agent 使用
        Args:
            stock_symbol (str): 股票代碼，例如 'AAPL', 'TSLA', 或台股 '2330'
        Returns:
            Tuple[bool, str]: 
            - 第一個值表示分析是否成功（True 為成功，False 表示發生錯誤或輸入無效）
            - 第二個值為分析結論或錯誤訊息（依 success 狀態而定）
        """

        # 若使用者未輸入任何代碼，直接提示錯誤（避免空白輸入導致 agent 無法運作）
        if not stock_symbol:
            return False,"請提供有效的股票代碼"
        
        # (台股)
        # ✅ 若輸入為純數字（如台股 '2330'），自動轉換為 Yahoo Finance 台股格式（加上 .TW）
        #    這是 Yahoo Finance 對台股的命名規則（如 2330 → 2330.TW）
        if stock_symbol.isdigit():
            stock_symbol = f"{stock_symbol}.TW"
        
        # (美股)
        # 格式化股票代碼為大寫(美股) => 去除首尾空白，並將股票代碼轉為大寫（確保符合 Yahoo Finance 的查詢標準）
        stock_symbol = stock_symbol.strip().upper()
        
        # ✅ 建立要傳送給 LLM 的提示語句（prompt），內容包含：
        #    - 查詢對象（股票代碼）
        #    - 分析方向（股價、財務數據、新聞、產業比較）
        #    - 請求明確的投資建議（買入、持有、賣出）並要求說明理由
        prompt = f"""
        分析 {stock_symbol} 股票是否值得購買。
        請考慮以下因素：
        當前股價
        公司基本資訊
        關鍵財務數據（如 P/E、P/B、股息收益率）
        分析師推薦
        最新相關新聞
        與行業平均水準的比較
        根據這些資訊，給出你的投資建議（買入、持有或賣出）並詳細解釋理由。 
        如果某些資訊無法取得，請在分析中說明，並基於可用資訊給出最佳判斷。
        """
        
        try:
            # ✅ 使用先前初始化好的 FunctionCallingAgent 來執行分析指令
            #    agent.chat(prompt) → 將 prompt 丟入 LLM，由模型自動呼叫必要的工具。
            #    在執行 agent.chat(prompt) 時，LLM 會根據 prompt 的語意推論是否需要呼叫某個工具，
            #    並主動產生對應的參數（如 symbol="AAPL"），再由 agent 自動呼叫對應的函式，並將結果回傳給 LLM，全流程由 LLM 自主判斷是否使用工具以及怎麼使用。   
            result = self.agent.chat(prompt)  # 🔁 呼叫會觸發相關 FunctionTool 以支援分析
            # 回傳最終模型分析結果（result.response 為LLM回應字串）
            return True,result.response
        
        # ⚠️ 若執行分析過程中出現錯誤（例如連線失敗、API 錯誤等），記錄到 log
        except Exception as e:
            logging.error(f"分析股票時出錯: {e}")
            # 回傳錯誤訊息給使用端
            return False,f"分析股票時出錯: {str(e)}"


def test():
    stock_service = StockService()
    success,result = stock_service.analyze_stock("2330")

    if success:
        print("✅ 測試成功，分析結果如下：")
        print(result)
    else:
        print("❌ 測試失敗，錯誤訊息：")
        print(result)



if __name__ == "__main__" :
    test()
