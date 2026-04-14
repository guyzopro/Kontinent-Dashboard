import pandas as pd
import numpy as np
import re
import io
import json
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import skew, kurtosis
import emoji

# --- 1. MOTEUR ANALYTIQUE ÉLITE (16 KPIs) ---

def calculate_elite_metrics(pnl_series, df_bot_f, df_act_f):
    if pnl_series.empty or len(pnl_series) < 2:
        return {}, pd.Series()
    
    returns = pnl_series.diff().fillna(0)
    total_pnl = pnl_series.iloc[-1]
    drawdown = pnl_series - pnl_series.cummax()
    max_dd = drawdown.min()
    
    # Calculs avancés
    std = returns.std()
    avg_ret = returns.mean()
    
    metrics = {
        "Net PNL": total_pnl,
        "Sharpe Ratio": (avg_ret / std * np.sqrt(len(returns))) if std != 0 else 0,
        "Max Drawdown": max_dd,
        "Profit Factor": (returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if returns[returns < 0].sum() != 0 else 0,
        "Win Rate %": (len(returns[returns > 0]) / len(returns[returns != 0]) * 100) if len(returns[returns != 0]) > 0 else 0,
        "Total Volume": df_bot_f['quantity'].sum() if not df_bot_f.empty else 0,
        "Calmar Ratio": abs(total_pnl / max_dd) if max_dd != 0 else 0,
        "Sortino Ratio": (avg_ret / returns[returns < 0].std() * np.sqrt(len(returns))) if not returns[returns < 0].empty and returns[returns < 0].std() != 0 else 0,
        "Recovery Factor": abs(total_pnl / max_dd) if max_dd != 0 else 0,
        "VaR (95%)": np.percentile(returns, 5),
        "Skewness": skew(returns),
        "Kurtosis": kurtosis(returns),
        "Total Trades": len(df_bot_f),
        "Avg Profit/Trade": total_pnl / len(df_bot_f) if len(df_bot_f) > 0 else 0,
        "Profit/1k Units": (total_pnl / (df_bot_f['quantity'].sum() / 1000)) if not df_bot_f.empty and df_bot_f['quantity'].sum() > 0 else 0,
        "Volatility (Daily)": std
    }

    # Calcul du Slippage moyen
    if not df_bot_f.empty and 'mid_price' in df_act_f.columns:
        df_mid = df_act_f[['timestamp', 'product', 'mid_price']].rename(columns={'product': 'symbol'})
        df_slip = pd.merge_asof(df_bot_f.sort_values('timestamp'), df_mid.sort_values('timestamp'), on='timestamp', by='symbol', direction='backward')
        metrics["Avg Slippage"] = abs(df_slip['price'] - df_slip['mid_price']).mean()
    else:
        metrics["Avg Slippage"] = 0

    return metrics, drawdown

# --- 2. PARSER UNIVERSEL (TEXTE + JSON) ---

def universal_prosperity_parser(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    df_act, df_tr = pd.DataFrame(), pd.DataFrame()

    if content.startswith('{'):
        try:
            data = json.loads(content)
            if "activitiesLog" in data:
                df_act = pd.read_csv(io.StringIO(data["activitiesLog"]), sep=";")
            if "tradeHistory" in data:
                raw_trades = data["tradeHistory"]
                df_tr = pd.DataFrame(json.loads(raw_trades) if isinstance(raw_trades, str) else raw_trades)
        except: pass
    else:
        act_match = re.search(r"Activities log:\s*(.*?)\s*Trade History:", content, re.DOTALL)
        if act_match:
            df_act = pd.read_csv(io.StringIO(act_match.group(1).strip()), sep=";")
        trade_pattern = re.compile(r'\{\s*"timestamp":\s*(\d+),.*?"buyer":\s*"([^"]*)",\s*"seller":\s*"([^"]*)",\s*"symbol":\s*"([^"]*)",.*?"price":\s*([\d\.]+),\s*"quantity":\s*(\d+),?\s*\}', re.DOTALL)
        df_tr = pd.DataFrame([{"timestamp": int(m[0]), "buyer": m[1], "seller": m[2], "symbol": m[3], "price": float(m[4]), "quantity": int(m[5])} for m in trade_pattern.findall(content)])

    if not df_tr.empty:
        df_tr['side'] = df_tr.apply(lambda x: 1 if x['buyer'] == 'SUBMISSION' else (-1 if x['seller'] == 'SUBMISSION' else 0), axis=1)
        df_bot = df_tr[df_tr['side'] != 0].copy()
        if not df_act.empty:
            df_bot['pos_change'] = df_bot['side'] * df_bot['quantity']
            df_bot = df_bot.sort_values(['symbol', 'timestamp'])
            df_bot['cum_pos'] = df_bot.groupby('symbol')['pos_change'].cumsum()
            pos_lookup = df_bot[['timestamp', 'symbol', 'cum_pos']].rename(columns={'symbol': 'product'})
            df_act = pd.merge_asof(df_act.sort_values('timestamp'), pos_lookup.sort_values('timestamp'), on='timestamp', by='product', direction='backward').fillna(0)
            df_act = df_act.rename(columns={'cum_pos': 'position'})
        return df_act, df_bot
    return df_act, pd.DataFrame()

# --- 3. DASHBOARD INTERFACE ---

def main():
    flag = emoji.emojize(":Cameroon:")
    st.set_page_config(page_title=f"Kontinent Terminal {flag}", layout="wide")
    st.title(f"🏛️ Kontinent Elite Terminal {flag}")

    uploaded_file = st.sidebar.file_uploader("Charger un log de trading", type=["txt", "log"])
    if not uploaded_file:
        st.info("Système en attente de données...")
        return

    with open("temp.txt", "wb") as f: f.write(uploaded_file.getbuffer())
    df_act, df_bot = universal_prosperity_parser("temp.txt")

    if df_act.empty:
        st.error("Impossible de parser le fichier.")
        return

    selected = st.sidebar.multiselect("Focus Instruments", options=sorted(df_act['product'].unique()), default=sorted(df_act['product'].unique()))
    df_act_f = df_act[df_act['product'].isin(selected)]
    df_bot_f = df_bot[df_bot['symbol'].isin(selected)] if not df_bot.empty else pd.DataFrame()

    # Calcul des KPIs
    pnl_series = df_act_f.groupby('timestamp')['profit_and_loss'].sum()
    metrics, dd_series = calculate_elite_metrics(pnl_series, df_bot_f, df_act_f)

    # Affichage des KPIs
    st.subheader("📊 Fleet Performance Metrics")
    m_cols = st.columns(5)
    for idx, (k, v) in enumerate(metrics.items()):
        m_cols[idx % 5].metric(k, f"{v:,.2f}" if abs(v) > 0.1 else f"{v:.4f}")

    # Onglets (Mise à jour avec l'onglet Volumes)
    t_pnl, t_dist, t_heat, t_corr, t_exec, t_inv, t_vol, t_tape = st.tabs([
        "📈 Equity", "📉 Stats", "🔥 Heatmap", "🔗 Correlation", "📊 Execution", "📦 Inventory", "📦 Volumes", "📜 Tape"
    ])

    with t_pnl:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pnl_series.index, y=pnl_series, name="PNL", line=dict(color='#00FFCC', width=3)))
        fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name="Drawdown", fill='tozeroy', line=dict(width=0), opacity=0.3))
        fig.update_layout(template="plotly_dark", height=500, title="Equity Curve vs Max Drawdown")
        st.plotly_chart(fig, use_container_width=True)

    with t_dist:
        rets = pnl_series.diff().fillna(0)
        fig_hist = px.histogram(rets, nbins=50, title="Distribution des Retours", color_discrete_sequence=['#00FFCC'], template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    with t_heat:
        df_act_f['time_step'] = pd.cut(df_act_f['timestamp'], bins=15, labels=False)
        heat_pivot = df_act_f.groupby(['product', 'time_step'])['profit_and_loss'].diff().dropna().groupby([df_act_f['product'], df_act_f['time_step']]).sum().unstack()
        fig_h = px.imshow(heat_pivot, color_continuous_scale='RdYlGn', title="Profitability Heatmap (Time vs Product)")
        fig_h.update_layout(template="plotly_dark")
        st.plotly_chart(fig_h, use_container_width=True)

    with t_corr:
        corr_matrix = df_act.pivot(index='timestamp', columns='product', values='profit_and_loss').diff().corr()
        st.plotly_chart(px.imshow(corr_matrix, text_auto=True, title="Matrice de Corrélation des Algos", template="plotly_dark"), use_container_width=True)

    with t_exec:
        vwap = []
        for s in selected:
            b = df_bot_f[(df_bot_f['symbol'] == s) & (df_bot_f['side'] == 1)]
            s_tr = df_bot_f[(df_bot_f['symbol'] == s) & (df_bot_f['side'] == -1)]
            vb = (b['price']*b['quantity']).sum()/b['quantity'].sum() if not b.empty and b['quantity'].sum() > 0 else 0
            vs = (s_tr['price']*s_tr['quantity']).sum()/s_tr['quantity'].sum() if not s_tr.empty and s_tr['quantity'].sum() > 0 else 0
            vwap.append({"Instrument": s, "VWAP Achat": round(vb,2), "VWAP Vente": round(vs,2), "Edge": round(vs-vb,2)})
        st.table(pd.DataFrame(vwap))

    with t_inv:
        st.plotly_chart(px.line(df_act_f, x='timestamp', y='position', color='product', title="Position Net Exposure", template="plotly_dark"), use_container_width=True)

    with t_vol:
        st.subheader("Analyse des Volumes par Produit et par Prix")
        
        if not df_bot_f.empty:
            col_v1, col_v2 = st.columns([1, 2])
            
            with col_v1:
                st.write("**Volume Total Tradé**")
                total_vol = df_bot_f.groupby('symbol')['quantity'].sum().reset_index()
                total_vol.columns = ['Produit', 'Volume Total']
                st.dataframe(total_vol, use_container_width=True, hide_index=True)

            with col_v2:
                fig_vol_tot = px.bar(total_vol, x='Produit', y='Volume Total', 
                                    title="Comparaison des Volumes",
                                    color='Produit', template="plotly_dark")
                st.plotly_chart(fig_vol_tot, use_container_width=True)

            st.divider()
            st.write("**Volume Acheté vs Vendu par Niveau de Prix (Market Profile)**")
            
            prod_select = st.selectbox("Sélectionner un produit pour le détail", options=selected)
            df_price_vol = df_bot_f[df_bot_f['symbol'] == prod_select].copy()
            
            price_volume = df_price_vol.groupby(['price', 'side'])['quantity'].sum().reset_index()
            price_volume['Type'] = price_volume['side'].map({1: 'Achat', -1: 'Vente'})
            
            fig_price = px.bar(price_volume, 
                               y="price", 
                               x="quantity", 
                               color="Type", 
                               orientation='h',
                               title=f"Volume at Price : {prod_select}",
                               color_discrete_map={'Achat': '#00FFCC', 'Vente': '#FF4B4B'},
                               labels={'price': 'Prix', 'quantity': 'Volume'},
                               template="plotly_dark")
            
            fig_price.update_layout(barmode='group', height=600)
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning("Aucune donnée de transaction disponible.")

    with t_tape:
        st.subheader("Journal de Transaction Submissions")
        st.dataframe(df_bot_f.sort_values('timestamp', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
