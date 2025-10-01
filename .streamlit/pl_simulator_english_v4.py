# -*- coding: utf-8 -*-
"""
P&L Simulation Web Interface
Based on Original Excel Structure

Date: 2025-10-01
Unit: Million KRW, %
Revenue Range: 160~360 Billion KRW (20 Billion intervals)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# Page Configuration
st.set_page_config(
    page_title="P&L Simulator",
    page_icon="📊",
    layout="wide"
)

# CSS Style
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-header">P&L Simulation</div>', unsafe_allow_html=True)
st.markdown("##### Based on Original Excel Structure | Unit: Million KRW, %")

# P&L Calculator Class
class PLCalculator:
    def __init__(self):
        # Default values (based on original Excel)
        self.avg_employees = 450  # Average employees
        self.a_commission = 5200  # A company commission (Million KRW)
        
        # Variable cost rates
        self.mc_rate = 0.57  # MC cost rate 57%
        self.material_consumption_rate = 0.0057  # Material consumption 0.57%
        self.transport_rate = 0.02  # Transportation 2%
        self.dev_supplies_rate = 0.012  # Development supplies 1.2%
        self.travel_rate = 0.02  # Travel expenses 2%
        self.commission_rate = 0.01  # Commission fee 1%
        self.other_expense_rate = 0.015  # Other expenses 1.5%
        
        # Fixed costs (Million KRW)
        self.inventory_disposal = 2000  # Inventory disposal
        self.salary_per_person = 7.31  # Salary per person (Million KRW/month)
        self.base_bonus = 300  # Performance bonus
        self.outsource_labor = 8800  # Outsource labor
        self.overseas_operation = 8400  # Overseas operation
        self.depreciation = 3168  # Depreciation
        self.welfare = 5.4  # Welfare per person (Million KRW/year)
        self.wip_diff = 0  # WIP difference
        
    def calculate_revenue_after_commission(self, revenue):
        """Revenue after commission deduction"""
        return revenue - self.a_commission
    
    def calculate_mc_cost(self, revenue):
        """Calculate MC cost"""
        # MC cost is calculated based on original revenue, not after commission
        return revenue * self.mc_rate
    
    def calculate_total_material_cost(self, revenue):
        """Calculate total material cost"""
        mc_cost = self.calculate_mc_cost(revenue)
        material_consumption = revenue * self.material_consumption_rate
        total = mc_cost + material_consumption + self.inventory_disposal
        return total
    
    def calculate_operating_expenses(self, revenue, target_profit_rate=None):
        """Calculate operating expenses"""
        # Personnel costs
        total_salary = self.avg_employees * self.salary_per_person * 12
        
        # Performance bonus
        # For 10% operating profit scenario: revenue * 0.011 + base_bonus
        # For normal scenario: base_bonus only
        if target_profit_rate == 0.1:
            bonus = revenue * 0.011 + self.base_bonus
        else:
            bonus = self.base_bonus
        
        # Variable costs
        transport = revenue * self.transport_rate
        dev_supplies = revenue * self.dev_supplies_rate
        travel = revenue * self.travel_rate
        commission = revenue * self.commission_rate
        other_expense = revenue * self.other_expense_rate
        
        # Welfare - annual welfare per person
        welfare_total = self.avg_employees * self.welfare
        
        total_operating = (total_salary + bonus + self.outsource_labor + 
                          self.overseas_operation + transport + dev_supplies +
                          travel + self.depreciation + welfare_total + 
                          commission + other_expense)
        
        return {
            'total_salary': total_salary,
            'bonus': bonus,
            'outsource_labor': self.outsource_labor,
            'overseas_operation': self.overseas_operation,
            'transport': transport,
            'dev_supplies': dev_supplies,
            'travel': travel,
            'depreciation': self.depreciation,
            'welfare': welfare_total,
            'commission': commission,
            'other_expense': other_expense,
            'total': total_operating
        }
    
    def calculate_operating_profit(self, revenue, target_profit_rate=None):
        """Calculate operating profit"""
        revenue_after_comm = self.calculate_revenue_after_commission(revenue)
        total_material = self.calculate_total_material_cost(revenue)
        operating_exp = self.calculate_operating_expenses(revenue, target_profit_rate)
        
        operating_profit = revenue_after_comm - total_material - operating_exp['total'] - self.wip_diff
        
        return {
            'revenue': revenue,
            'revenue_after_commission': revenue_after_comm,
            'total_material_cost': total_material,
            'mc_cost': self.calculate_mc_cost(revenue),
            'operating_expenses': operating_exp['total'],
            'operating_profit': operating_profit,
            'operating_margin': (operating_profit / revenue * 100) if revenue > 0 else 0,
            'mc_cost_rate': (self.calculate_mc_cost(revenue) / revenue * 100) if revenue > 0 else 0
        }
    
    def find_bep_mc_rate(self, revenue):
        """Calculate MC cost rate for Break-Even Point (BEP)"""
        revenue_after_comm = self.calculate_revenue_after_commission(revenue)
        operating_exp = self.calculate_operating_expenses(revenue, target_profit_rate=None)
        
        # Calculate MC cost for operating profit = 0
        other_material = revenue * self.material_consumption_rate + self.inventory_disposal
        required_mc_cost = revenue_after_comm - other_material - operating_exp['total']
        required_mc_rate = (required_mc_cost / revenue * 100) if revenue > 0 else 0
        
        return required_mc_rate
    
    def find_profit10_mc_rate(self, revenue):
        """Calculate MC cost rate for 10% Operating Profit"""
        revenue_after_comm = self.calculate_revenue_after_commission(revenue)
        
        # Target operating profit: 10% of revenue
        target_operating_profit = revenue * 0.10
        
        # Operating expenses with 1.1% bonus for 10% profit scenario
        operating_exp = self.calculate_operating_expenses(revenue, target_profit_rate=0.1)
        
        # Calculate required MC cost
        # Formula: Revenue After Comm - Total Material - Operating Exp - WIP = Operating Profit
        # Rearrange: Total Material = Revenue After Comm - Operating Exp - WIP - Operating Profit
        other_material = revenue * self.material_consumption_rate + self.inventory_disposal
        required_total_material = revenue_after_comm - operating_exp['total'] - self.wip_diff - target_operating_profit
        required_mc_cost = required_total_material - other_material
        required_mc_rate = (required_mc_cost / revenue * 100) if revenue > 0 else 0
        
        return required_mc_rate

# Create calculator instance
calculator = PLCalculator()

# Sidebar - Parameter Settings
st.sidebar.header("PL Simulator")

# Basic Information
st.sidebar.subheader("Basic Info")
calculator.avg_employees = st.sidebar.number_input("Average Employees", min_value=1, value=450, step=10)
calculator.a_commission = st.sidebar.number_input("A Company Fee (Million KRW)", min_value=0, value=5200, step=100)

# Variable Cost Rates
st.sidebar.subheader("Variable Cost (%)")
calculator.mc_rate = st.sidebar.slider("MC Cost Rate", min_value=0.0, max_value=100.0, value=57.0, step=0.1) / 100
calculator.material_consumption_rate = st.sidebar.slider("Material Consumption", min_value=0.0, max_value=10.0, value=0.57, step=0.01) / 100
calculator.transport_rate = st.sidebar.slider("Transportation", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
calculator.dev_supplies_rate = st.sidebar.slider("Development Supplies", min_value=0.0, max_value=10.0, value=1.2, step=0.1) / 100
calculator.travel_rate = st.sidebar.slider("Travel Expenses", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100
calculator.commission_rate = st.sidebar.slider("Commission Fee", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100
calculator.other_expense_rate = st.sidebar.slider("Other Expenses", min_value=0.0, max_value=10.0, value=1.5, step=0.1) / 100

# Fixed Costs
st.sidebar.subheader("Fixed Cost (Million KRW)")
calculator.inventory_disposal = st.sidebar.number_input("Inventory Disposal", min_value=0, value=2000, step=100)
calculator.salary_per_person = st.sidebar.number_input("Monthly Salary per Person", min_value=0.0, value=7.31, step=0.1)
calculator.base_bonus = st.sidebar.number_input("Performance Bonus", min_value=0, value=300, step=100)
calculator.outsource_labor = st.sidebar.number_input("Outsource Labor", min_value=0, value=8800, step=100)
calculator.overseas_operation = st.sidebar.number_input("Overseas Operation", min_value=0, value=8400, step=100)
calculator.depreciation = st.sidebar.number_input("Depreciation", min_value=0, value=3168, step=100)
calculator.welfare = st.sidebar.number_input("Annual Welfare per Person", min_value=0.0, value=5.4, step=0.1)

# Revenue Settings
st.sidebar.subheader("Revenue Settings")

# Base Revenue for Current P&L analysis
base_revenue = st.sidebar.number_input(
    "Base Revenue (Million KRW)", 
    min_value=50000, 
    value=200000, 
    step=10000,
    help="This will be the default revenue for Current P&L analysis"
)

st.sidebar.markdown("---")

# Simulation Range - automatically calculated based on Base Revenue
st.sidebar.markdown("**Simulation Range**")
min_revenue = base_revenue - 100000
max_revenue = base_revenue + 100000

# Display the auto-calculated range
col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.text(f"Min: {min_revenue:,}")
with col2:
    st.sidebar.text(f"Max: {max_revenue:,}")

step_size = st.sidebar.number_input("Interval (Million KRW)", min_value=10000, value=20000, step=10000)

# Run Simulation Button
run_simulation = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

# Main Content
if run_simulation:
    with st.spinner('Calculating simulation...'):
        # Generate revenue range
        revenue_range = np.arange(min_revenue, max_revenue + step_size, step_size)
        
        # Calculate scenarios
        normal_results = []
        bep_results = []
        profit10_results = []
        
        for revenue in revenue_range:
            # Normal scenario
            result_normal = calculator.calculate_operating_profit(revenue)
            normal_results.append(result_normal)
            
            # BEP scenario - Calculate MC cost rate
            bep_mc_rate = calculator.find_bep_mc_rate(revenue)
            bep_results.append({
                'revenue': revenue,
                'required_mc_rate': bep_mc_rate,
                'mc_cost': revenue * bep_mc_rate / 100
            })
            
            # Operating profit 10% scenario - Calculate required MC cost rate
            profit10_mc_rate = calculator.find_profit10_mc_rate(revenue)
            profit10_mc_cost = revenue * profit10_mc_rate / 100
            
            # Calculate actual profit with this MC rate
            revenue_after_comm = calculator.calculate_revenue_after_commission(revenue)
            material_consumption = revenue * calculator.material_consumption_rate
            total_material = profit10_mc_cost + material_consumption + calculator.inventory_disposal
            operating_exp = calculator.calculate_operating_expenses(revenue, target_profit_rate=0.1)
            operating_profit = revenue_after_comm - total_material - operating_exp['total'] - calculator.wip_diff
            
            profit10_results.append({
                'revenue': revenue,
                'revenue_after_commission': revenue_after_comm,
                'total_material_cost': total_material,
                'mc_cost': profit10_mc_cost,
                'operating_expenses': operating_exp['total'],
                'operating_profit': operating_profit,
                'operating_margin': (operating_profit / revenue * 100) if revenue > 0 else 0,
                'mc_cost_rate': profit10_mc_rate
            })
        
        # Create dataframes
        df_normal = pd.DataFrame(normal_results)
        df_bep = pd.DataFrame(bep_results)
        df_profit10 = pd.DataFrame(profit10_results)
        
        # Save to session
        st.session_state['df_normal'] = df_normal
        st.session_state['df_bep'] = df_bep
        st.session_state['df_profit10'] = df_profit10
        st.session_state['simulation_done'] = True
    
    st.success("Simulation completed!")

# Display Results
if 'simulation_done' in st.session_state and st.session_state['simulation_done']:
    df_normal = st.session_state['df_normal']
    df_bep = st.session_state['df_bep']
    df_profit10 = st.session_state['df_profit10']
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Current P&L", "Break-Even Point", "Operating Profit 10%", "Download"])
    
    # Tab 1: Current P&L Scenario - DETAILED P&L STATEMENT
    with tab1:
        st.subheader("■ Simulation Result")
        st.caption("Unit: Million KRW, %")
        
        # Find the index of base_revenue in the dataframe
        try:
            base_revenue_index = int(df_normal[df_normal['revenue'] == base_revenue].index[0])
        except:
            base_revenue_index = len(df_normal)//2  # Fallback to middle if base_revenue not found
        
        # Revenue selector
        selected_revenue = st.selectbox(
            "Select Revenue to View Detailed P&L",
            options=df_normal['revenue'].tolist(),
            format_func=lambda x: f"{x:,.0f} Million KRW ({x/1000:,.0f} Billion KRW)",
            index=base_revenue_index  # Default to base_revenue
        )
        
        # Get selected revenue data
        selected_data = df_normal[df_normal['revenue'] == selected_revenue].iloc[0]
        
        # Calculate detailed breakdown
        revenue = selected_data['revenue']
        revenue_after_comm = selected_data['revenue_after_commission']
        mc_cost = selected_data['mc_cost']
        material_consumption = revenue * calculator.material_consumption_rate
        total_material = selected_data['total_material_cost']
        
        # Operating expenses breakdown
        operating_exp = calculator.calculate_operating_expenses(revenue)
        
        # Create detailed P&L statement
        st.markdown("---")
        
        # Summary metrics at top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Employees", f"{calculator.avg_employees:,.0f}")
        with col2:
            st.metric("Operating Profit", f"{selected_data['operating_profit']:,.0f}M", 
                     delta=f"{selected_data['operating_margin']:.2f}%")
        with col3:
            st.metric("MC Cost Rate", f"{selected_data['mc_cost_rate']:.2f}%")
        with col4:
            st.metric("Revenue (after commission)", f"{revenue_after_comm:,.0f}M")
        
        st.markdown("---")
        
        # Detailed P&L Statement in table format
        pl_data = {
            'Category': [
                'I. Revenue (After Commission)',
                '  ├─ Revenue',
                '  └─ A Company Commission',
                'II. Total Material Cost',
                '  ├─ MC Cost',
                '  ├─ Material Consumption',
                '  └─ Inventory Disposal',
                'III. Operating Expenses',
                '  ├─ Salary (including allowances)',
                '  ├─ Performance Bonus',
                '  ├─ Outsource Labor',
                '  ├─ Overseas Operation',
                '  ├─ Transportation',
                '  ├─ Development Supplies',
                '  ├─ Travel Expenses',
                '  ├─ Depreciation',
                '  ├─ Welfare',
                '  ├─ Commission Fee',
                '  └─ Other Expenses',
                'IV. WIP Difference',
                'V. Operating Profit'
            ],
            'Amount (M KRW)': [
                revenue_after_comm,
                revenue,
                -calculator.a_commission,
                total_material,
                mc_cost,
                material_consumption,
                calculator.inventory_disposal,
                operating_exp['total'],
                operating_exp['total_salary'],
                operating_exp['bonus'],
                operating_exp['outsource_labor'],
                operating_exp['overseas_operation'],
                operating_exp['transport'],
                operating_exp['dev_supplies'],
                operating_exp['travel'],
                operating_exp['depreciation'],
                operating_exp['welfare'],
                operating_exp['commission'],
                operating_exp['other_expense'],
                calculator.wip_diff,
                selected_data['operating_profit']
            ],
            'Ratio (%)': [
                revenue_after_comm / revenue * 100,
                100.0,
                -calculator.a_commission / revenue * 100,
                total_material / revenue * 100,
                mc_cost / revenue * 100,
                material_consumption / revenue * 100,
                calculator.inventory_disposal / revenue * 100,
                operating_exp['total'] / revenue * 100,
                operating_exp['total_salary'] / revenue * 100,
                operating_exp['bonus'] / revenue * 100,
                operating_exp['outsource_labor'] / revenue * 100,
                operating_exp['overseas_operation'] / revenue * 100,
                operating_exp['transport'] / revenue * 100,
                operating_exp['dev_supplies'] / revenue * 100,
                operating_exp['travel'] / revenue * 100,
                operating_exp['depreciation'] / revenue * 100,
                operating_exp['welfare'] / revenue * 100,
                operating_exp['commission'] / revenue * 100,
                operating_exp['other_expense'] / revenue * 100,
                calculator.wip_diff / revenue * 100,
                selected_data['operating_profit'] / revenue * 100
            ]
        }
        
        df_pl = pd.DataFrame(pl_data)
        
        # Format the dataframe for display
        df_pl_display = df_pl.copy()
        df_pl_display['Amount (M KRW)'] = df_pl_display['Amount (M KRW)'].apply(lambda x: f"{x:,.2f}")
        df_pl_display['Ratio (%)'] = df_pl_display['Ratio (%)'].apply(lambda x: f"{x:.2f}")
        
        # Apply styling to highlight main categories
        def highlight_main_categories(row):
            if row['Category'].startswith('I.') or row['Category'].startswith('II.') or \
               row['Category'].startswith('III.') or row['Category'].startswith('IV.') or \
               row['Category'].startswith('V.'):
                return ['background-color: #808080; font-weight: bold; color: white'] * len(row)
            return [''] * len(row)
        
        styled_df = df_pl_display.style.apply(highlight_main_categories, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=800)
        
        st.markdown("---")
        
        # Visualization section
        st.subheader("Cost Structure Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for cost breakdown
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Total Material Cost', 'Operating Expenses', 'Operating Profit'],
                values=[total_material, operating_exp['total'], max(0, selected_data['operating_profit'])],
                hole=.3,
                marker_colors=['#ff9999', '#66b3ff', '#99ff99']
            )])
            fig_pie.update_layout(title='Cost Structure', height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Operating expenses breakdown
            expense_labels = ['Salary', 'Bonus', 'Outsource', 'Overseas', 'Transport', 
                            'Dev Supplies', 'Travel', 'Depreciation', 'Welfare', 'Commission', 'Other']
            expense_values = [
                operating_exp['total_salary'],
                operating_exp['bonus'],
                operating_exp['outsource_labor'],
                operating_exp['overseas_operation'],
                operating_exp['transport'],
                operating_exp['dev_supplies'],
                operating_exp['travel'],
                operating_exp['depreciation'],
                operating_exp['welfare'],
                operating_exp['commission'],
                operating_exp['other_expense']
            ]
            
            fig_expenses = go.Figure(data=[go.Bar(
                x=expense_labels,
                y=expense_values,
                marker_color='lightblue'
            )])
            fig_expenses.update_layout(
                title='Operating Expenses Breakdown',
                xaxis_title='Category',
                yaxis_title='Amount (M KRW)',
                height=400,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_expenses, use_container_width=True)
        
        # Multi-revenue comparison chart
        st.markdown("---")
        st.subheader("Multi-Revenue Scenario Comparison")
        
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Operating Profit Trend', 'MC Cost & MC Cost Rate', 
                          'Cost Structure', 'Operating Margin'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Operating profit trend
        fig_multi.add_trace(
            go.Scatter(x=df_normal['revenue']/1000, y=df_normal['operating_profit'],
                      name='Operating Profit', line=dict(color='green', width=3)),
            row=1, col=1
        )
        fig_multi.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # MC cost and rate
        fig_multi.add_trace(
            go.Bar(x=df_normal['revenue']/1000, y=df_normal['mc_cost'],
                  name='MC Cost', marker_color='blue'),
            row=1, col=2, secondary_y=False
        )
        fig_multi.add_trace(
            go.Scatter(x=df_normal['revenue']/1000, y=df_normal['mc_cost_rate'],
                      name='MC Cost Rate', line=dict(color='red', width=2)),
            row=1, col=2, secondary_y=True
        )
        
        # Cost structure
        fig_multi.add_trace(
            go.Scatter(x=df_normal['revenue']/1000, y=df_normal['total_material_cost'],
                      name='Total Material Cost', fill='tozeroy', fillcolor='rgba(255,153,153,0.5)'),
            row=2, col=1
        )
        fig_multi.add_trace(
            go.Scatter(x=df_normal['revenue']/1000, y=df_normal['operating_expenses'],
                      name='Operating Expenses', fill='tozeroy', fillcolor='rgba(102,179,255,0.5)'),
            row=2, col=1
        )
        
        # Operating margin
        fig_multi.add_trace(
            go.Scatter(x=df_normal['revenue']/1000, y=df_normal['operating_margin'],
                      name='Operating Margin', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        fig_multi.update_xaxes(title_text="Revenue (Billion KRW)", row=1, col=1)
        fig_multi.update_xaxes(title_text="Revenue (Billion KRW)", row=1, col=2)
        fig_multi.update_xaxes(title_text="Revenue (Billion KRW)", row=2, col=1)
        fig_multi.update_xaxes(title_text="Revenue (Billion KRW)", row=2, col=2)
        
        fig_multi.update_yaxes(title_text="Operating Profit (M KRW)", row=1, col=1)
        fig_multi.update_yaxes(title_text="MC Cost (M KRW)", row=1, col=2, secondary_y=False)
        fig_multi.update_yaxes(title_text="MC Cost Rate (%)", row=1, col=2, secondary_y=True)
        fig_multi.update_yaxes(title_text="Cost (M KRW)", row=2, col=1)
        fig_multi.update_yaxes(title_text="Operating Margin (%)", row=2, col=2)
        
        fig_multi.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig_multi, use_container_width=True)
    
    # Tab 2: BEP Scenario
    with tab2:
        st.subheader("MC Cost Rate Required for Break-Even Point (BEP)")
        st.info("Calculates the MC cost rate needed to achieve operating profit of 0 at each revenue level.")
        
        # Chart
        fig_bep = go.Figure()
        
        fig_bep.add_trace(go.Bar(
            x=df_bep['revenue']/1000,
            y=df_bep['mc_cost'],
            name='Required MC Cost',
            marker_color='blue',
            text=df_bep['mc_cost'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside'
        ))
        
        fig_bep.add_trace(go.Scatter(
            x=df_bep['revenue']/1000,
            y=df_bep['required_mc_rate'],
            name='Required MC Cost Rate',
            yaxis='y2',
            line=dict(color='red', width=3),
            text=df_bep['required_mc_rate'].apply(lambda x: f'{x:.2f}%'),
            textposition='top center',
            mode='lines+markers+text'
        ))
        
        # Calculate max values for proper scaling
        max_mc_cost = df_bep['mc_cost'].max()
        max_mc_rate = df_bep['required_mc_rate'].max()
        
        fig_bep.update_layout(
            title='MC Cost & MC Cost Rate for BEP Achievement',
            xaxis_title='Revenue (Billion KRW)',
            yaxis_title='MC Cost (M KRW)',
            yaxis=dict(range=[0, max_mc_cost * 1.5]),  # Primary axis: 0 to 1.5x max
            yaxis2=dict(
                title='MC Cost Rate (%)', 
                overlaying='y', 
                side='right',
                range=[0, max(100, max_mc_rate * 1.2)]  # Secondary axis: 0 to max or 100%
            ),
            xaxis=dict(
                tickmode='linear',
                tick0=min_revenue/1000,
                dtick=step_size/1000
            ),
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_bep, use_container_width=True)
        
        # Detailed data
        st.subheader("BEP Detailed Data")
        df_bep_display = df_bep.copy()
        df_bep_display['revenue'] = (df_bep_display['revenue']/1000).apply(lambda x: f"{x:,.0f}")
        df_bep_display['mc_cost'] = df_bep_display['mc_cost'].apply(lambda x: f"{x:,.0f}")
        df_bep_display['required_mc_rate'] = df_bep_display['required_mc_rate'].apply(lambda x: f"{x:.2f}%")
        df_bep_display.columns = ['Revenue(Billion)', 'Required MC Rate', 'Required MC Cost']
        st.dataframe(df_bep_display, use_container_width=True)
    
    # Tab 3: Operating Profit 10% Scenario
    with tab3:
        st.subheader("Operating Profit 10% Achievement Scenario")
        st.info("Calculates MC cost rate to achieve 10% operating profit of revenue. (Performance bonus: 1.1% of revenue)")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_mc_rate_p10 = df_profit10['mc_cost_rate'].mean()
            st.metric("Avg Required MC Rate", f"{avg_mc_rate_p10:.2f}%")
        
        with col2:
            total_profit = df_profit10['operating_profit'].sum()
            st.metric("Total Expected Profit", f"{total_profit:,.0f}M")
        
        with col3:
            current_mc_rate = calculator.mc_rate * 100
            st.metric("Current MC Rate", f"{current_mc_rate:.2f}%")
        
        # Chart
        fig_p10 = go.Figure()
        
        fig_p10.add_trace(go.Bar(
            x=df_profit10['revenue']/1000,
            y=df_profit10['mc_cost'],
            name='MC Cost',
            marker_color='blue',
            text=df_profit10['mc_cost'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside'
        ))
        
        fig_p10.add_trace(go.Scatter(
            x=df_profit10['revenue']/1000,
            y=df_profit10['mc_cost_rate'],
            name='MC Cost Rate',
            yaxis='y2',
            line=dict(color='red', width=3),
            text=df_profit10['mc_cost_rate'].apply(lambda x: f'{x:.2f}%'),
            textposition='top center',
            mode='lines+markers+text'
        ))
        
        # Calculate max values for proper scaling
        max_mc_cost_p10 = df_profit10['mc_cost'].max()
        max_mc_rate_p10 = df_profit10['mc_cost_rate'].max()
        
        fig_p10.update_layout(
            title='MC Cost & MC Cost Rate for 10% Operating Profit',
            xaxis_title='Revenue (Billion KRW)',
            yaxis_title='MC Cost (M KRW)',
            yaxis=dict(range=[0, max_mc_cost_p10 * 1.5]),  # Primary axis: 0 to 1.5x max
            yaxis2=dict(
                title='MC Cost Rate (%)', 
                overlaying='y', 
                side='right',
                range=[0, max(100, max_mc_rate_p10 * 1.2)]  # Secondary axis: 0 to max or 100%
            ),
            xaxis=dict(
                tickmode='linear',
                tick0=min_revenue/1000,
                dtick=step_size/1000
            ),
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_p10, use_container_width=True)
        
        # Detailed data
        st.subheader("Operating Profit 10% Detailed Data")
        df_p10_display = df_profit10.copy()
        df_p10_display['revenue'] = (df_p10_display['revenue']/1000).apply(lambda x: f"{x:,.0f}")
        df_p10_display['mc_cost'] = df_p10_display['mc_cost'].apply(lambda x: f"{x:,.0f}")
        df_p10_display['operating_profit'] = df_p10_display['operating_profit'].apply(lambda x: f"{x:,.0f}")
        df_p10_display['mc_cost_rate'] = df_p10_display['mc_cost_rate'].apply(lambda x: f"{x:.2f}%")
        df_p10_display.columns = ['Revenue(Billion)', 'After Commission', 'Total Material', 'MC Cost', 'Operating Exp', 'Operating Profit', 'Operating Margin', 'MC Rate']
        st.dataframe(df_p10_display[['Revenue(Billion)', 'MC Cost', 'MC Rate', 'Operating Profit', 'Operating Margin']], use_container_width=True)
    
    # Tab 4: Download
    with tab4:
        st.subheader("Download Results")
        
        # Excel file generation
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Current P&L
            df_normal.to_excel(writer, sheet_name='Current_PL', index=False)
            # BEP
            df_bep.to_excel(writer, sheet_name='BEP_Scenario', index=False)
            # Operating profit 10%
            df_profit10.to_excel(writer, sheet_name='Profit10_Scenario', index=False)
        
        output.seek(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download Full Results (Excel)",
                data=output.getvalue(),
                file_name=f"PL_Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # CSV download
            csv_normal = df_normal.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download Current P&L (CSV)",
                data=csv_normal,
                file_name=f"PL_Current_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.success("Download the file in your preferred format!")

else:
    # Information before simulation
    st.info("""
    ### How to Use P&L Simulation
    
    1. Set parameters in the left sidebar
    2. Click the "Run Simulation" button
    3. Review results and download Excel/CSV files
    
    #### Provided Scenarios:
    - **Current P&L**: P&L calculation based on current MC cost rate
    - **BEP (Break-Even Point)**: Required MC cost rate for operating profit = 0
    - **Operating Profit 10%**: Required MC cost rate to achieve 10% operating profit
    
    #### Key Features:
    - Same calculation logic as original Excel file
    - Revenue range: 160~360 billion KRW (default, adjustable)
    - Unit: Million KRW, %
    """)
    
    st.warning("Set parameters in the left sidebar and start simulation!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
P&L Simulation | Based on Original Excel Structure | 2025-10-01
</div>
""", unsafe_allow_html=True)
