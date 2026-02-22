// ══════════════════════════════════════════════════════════════
// WEATHER TRADER BOT — Google Apps Script
// 
// SETUP:
// 1. Create a Google Sheet called "Weather Trader Log"
// 2. Go to Extensions → Apps Script
// 3. Paste this entire code, replacing any existing code
// 4. Click Deploy → New Deployment → Web App
//    - Execute as: Me
//    - Who has access: Anyone
// 5. Click Deploy, authorize when prompted
// 6. Copy the Web App URL → paste into your .env as GOOGLE_SHEET_WEBHOOK
// ══════════════════════════════════════════════════════════════

function doPost(e) {
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet();
    var data = JSON.parse(e.postData.contents);
    
    // Determine which sheet tab to use
    var eventType = data.event || "cycle";
    var tabName = eventType === "resolution" ? "Resolutions" 
                : eventType === "daily_summary" ? "Summaries" 
                : "Cycles";
    
    var tab = sheet.getSheetByName(tabName);
    if (!tab) {
      tab = sheet.insertSheet(tabName);
      // Add headers on first use
      if (tabName === "Cycles") {
        tab.appendRow([
          "Timestamp", "City", "Market Date", "Ensemble Mean", "Members",
          "METAR High", "Brackets", "Signals", "Trades",
          "B0 Label", "B0 Corr%", "B0 Mkt%", "B0 Edge", "B0 Signal",
          "B1 Label", "B1 Corr%", "B1 Mkt%", "B1 Edge", "B1 Signal",
          "B2 Label", "B2 Corr%", "B2 Mkt%", "B2 Edge", "B2 Signal",
          "B3 Label", "B3 Corr%", "B3 Mkt%", "B3 Edge", "B3 Signal",
          "B4 Label", "B4 Corr%", "B4 Mkt%", "B4 Edge", "B4 Signal",
          "B5 Label", "B5 Corr%", "B5 Mkt%", "B5 Edge", "B5 Signal",
          "B6 Label", "B6 Corr%", "B6 Mkt%", "B6 Edge", "B6 Signal",
          "B7 Label", "B7 Corr%", "B7 Mkt%", "B7 Edge", "B7 Signal",
          "B8 Label", "B8 Corr%", "B8 Mkt%", "B8 Edge", "B8 Signal",
          "T0 Bracket", "T0 Ask", "T0 Bet", "T0 Edge", "T0 OrderID",
          "T1 Bracket", "T1 Ask", "T1 Bet", "T1 Edge", "T1 OrderID",
          "T2 Bracket", "T2 Ask", "T2 Bet", "T2 Edge", "T2 OrderID",
        ]);
        tab.getRange(1, 1, 1, tab.getLastColumn()).setFontWeight("bold");
      } else if (tabName === "Resolutions") {
        tab.appendRow([
          "Timestamp", "City", "Market Date", "Winner", "P&L", "Positions"
        ]);
        tab.getRange(1, 1, 1, tab.getLastColumn()).setFontWeight("bold");
      } else if (tabName === "Summaries") {
        tab.appendRow([
          "Timestamp", "Active Positions", "Active Exposure",
          "Total Trades", "Wins", "Losses", "Win Rate",
          "Total Wagered", "Total P&L", "ROI%"
        ]);
        tab.getRange(1, 1, 1, tab.getLastColumn()).setFontWeight("bold");
      }
    }
    
    // Build row based on tab type
    var row = [];
    if (tabName === "Cycles") {
      row = [
        data.timestamp, data.city, data.market_date,
        data.ensemble_mean, data.ensemble_members,
        data.metar_day_high, data.num_brackets, data.num_signals, data.num_trades,
      ];
      // Add bracket data (up to 9)
      for (var i = 0; i < 9; i++) {
        var p = "b" + i;
        row.push(data[p + "_label"] || "");
        row.push(data[p + "_corr"] || "");
        row.push(data[p + "_mkt"] || "");
        row.push(data[p + "_edge"] || "");
        row.push(data[p + "_signal"] || "");
      }
      // Add trade data (up to 3)
      for (var i = 0; i < 3; i++) {
        var p = "t" + i;
        row.push(data[p + "_bracket"] || "");
        row.push(data[p + "_ask"] || "");
        row.push(data[p + "_bet"] || "");
        row.push(data[p + "_edge"] || "");
        row.push(data[p + "_order_id"] || "");
      }
    } else if (tabName === "Resolutions") {
      row = [
        data.timestamp, data.city, data.market_date,
        data.winner, data.pnl, data.positions_resolved,
      ];
    } else if (tabName === "Summaries") {
      row = [
        data.timestamp, data.active_positions, data.active_exposure,
        data.total_trades, data.wins, data.losses, data.win_rate,
        data.total_wagered, data.total_pnl, data.roi,
      ];
    }
    
    tab.appendRow(row);
    
    return ContentService.createTextOutput(
      JSON.stringify({status: "ok"})
    ).setMimeType(ContentService.MimeType.JSON);
    
  } catch (err) {
    return ContentService.createTextOutput(
      JSON.stringify({status: "error", message: err.toString()})
    ).setMimeType(ContentService.MimeType.JSON);
  }
}

// Test function — run manually to verify setup
function testWebhook() {
  var e = {
    postData: {
      contents: JSON.stringify({
        timestamp: new Date().toISOString(),
        city: "London",
        market_date: "2026-02-22",
        ensemble_mean: 12.3,
        ensemble_members: 176,
        metar_day_high: 13,
        num_brackets: 9,
        num_signals: 1,
        num_trades: 0,
        b0_label: "12°C",
        b0_corr: 25.3,
        b0_mkt: 18.0,
        b0_edge: 4.2,
        b0_signal: "FILTERED",
      })
    }
  };
  var result = doPost(e);
  Logger.log(result.getContent());
}
