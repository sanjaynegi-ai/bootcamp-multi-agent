from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool
from typing import List, Optional
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


class MarketSignal(BaseModel):
    """Represents a company currently highlighted in financial news"""
    name: str = Field(..., description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol, if available")
    reason: str = Field(..., description="News driver or event making the company trend")

class MarketSignalList(BaseModel):
    """Collection of companies identified as current market signals"""
    signals: List[MarketSignal] = Field(..., description="List of companies trending in the sector")

class CompanyResearch(BaseModel):
    """Structured research profile for a single company"""
    name: str = Field(..., description="Company name")
    market_position: str = Field(..., description="Competitive standing and industry role")
    financials: Optional[str] = Field(None, description="Key financial metrics and valuation insights")
    future_outlook: str = Field(..., description="Projected growth trajectory and strategic outlook")
    investment_potential: str = Field(..., description="Assessment of suitability as an investment candidate")
    risks: Optional[str] = Field(None, description="Key risks or challenges impacting the company")

class CompanyResearchBundleList(BaseModel):
    """Aggregated research reports for all candidate companies"""
    reports: List[CompanyResearch] = Field(..., description="Detailed research collection for trending companies")

@CrewBase
class StockFinalizer():
    """StockFinalizer crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def trend_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['trend_scout'], # type: ignore[index]
            tools=[SerperDevTool()], 
            memory=True,
            verbose=True
        )

    @agent
    def equity_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['equity_analyst'], # type: ignore[index]
            tools=[SerperDevTool()], 
            memory=True,
            verbose=True
        )

    @agent
    def investment_selector(self) -> Agent:
        return Agent(
            config=self.agents_config['investment_selector'], # type: ignore[index]\
            memory=True,
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def discover_trends(self) -> Task:
        return Task(
            config=self.tasks_config['discover_trends'], # type: ignore[index]
            output_pydantic = MarketSignalList
        )

    @task
    def analyze_companies(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_companies'], # type: ignore[index]
            output_pydantic = CompanyResearchBundleList
        )

    @task
    def select_investment(self) -> Task:
        return Task(
            config=self.tasks_config['select_investment'], # type: ignore[index]
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the StockFinalizer crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        portfolio_manager = Agent(
            config=self.agents_config['portfolio_manager'],
            allow_delegation=True
        )

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
            verbose=True,
            tracing=True,
            memory=True,
            manager_agent=portfolio_manager
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
