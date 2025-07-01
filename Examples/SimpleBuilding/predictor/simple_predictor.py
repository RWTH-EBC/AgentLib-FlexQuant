import agentlib as al
import numpy as np
import pandas as pd
from typing import List
import sys
from pydantic import BaseModel, Field
from Examples.SimpleBuilding.utils.day_of_year import get_day_of_year
# from agentlib.core.datamodels import FlexEventVariable
from typing import Optional
from Examples.SimpleBuilding.utils.schedules import SCHEDULES
from Examples.SimpleBuilding.utils.temp_traj_adapter import temp_trajectory_adapter, temp_shift_traj_adapter


class Schedule(BaseModel):
    data: dict = Field(default={})
    type: str = Field(default="default")
    """Schedule for the day, with start and end times for each comfort mode-"""

    @staticmethod
    def get_schedule_by_type(schedule_type: str) -> dict:
        return SCHEDULES.get(schedule_type.lower(), {})
        

class WoliszSchedule(Schedule):
    type: str = "wolisz"
    data: dict = Schedule.get_schedule_by_type("wolisz")

class SimpleSchedule(Schedule):
    type: str = "simple"
    data: dict =  Schedule.get_schedule_by_type("simple")

class TestSchedule(Schedule):
    type: str = "test"
    data: dict = Schedule.get_schedule_by_type("test")

class FlexEvent(BaseModel):
    upper_boundary_shift: Optional[float] = Field(default=None, metadata={"title": "Upper boundary shift"})
    lower_boundary_shift: Optional[float] = Field(default=None, metadata={"title": "Lower boundary shift"})
    gradient: Optional[float] = Field(default=None, metadata={"title": "Gradient"})
    start: Optional[int] = Field(default=None, metadata={"title": "Start time (s)"})
    end: Optional[int] = Field(default=None, metadata={"title": "End time (s)"})

    def as_dict(self):
        return self.model_dump() 


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the ambient temp and comfort setpoint
    at a specified interval"""

    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="T_amb", 
            type="pd.Series", 
            description="Ambient air temperature",
        ),
        al.AgentVariable(
            name="T_upper", 
            type="pd.Series", 
            description="Upper boundary (soft) for T",
        ),
        al.AgentVariable(
            name="T_lower",
            type="pd.Series", 
            description="Lower boundary (soft) for T",
        ),
        al.AgentVariable(
            name="r_pel", 
            unit="ct/kWh", 
            type="pd.Series", 
            description="Weight for P_el in objective function (electricity price)"
        ),
        al.AgentVariable(
            name="T_upper_shadow",
            alias="T_upper_shadow",
            description="Upper Temperature during Flexevent for the shadow MPCs'",
        ),
        al.AgentVariable(
            name="T_lower_shadow", 
            alias="T_lower_shadow",
            description="Lower Temperature during Flexevent for the shadow MPCs'",
        ),
    ]

    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="prediction_sampling_time", 
            value=10, 
            description="Sampling time for prediction.",
        ),
        al.AgentVariable(
            name="prediction_horizon",
            value=10,
            description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="sampling_time",
            value=900,
            description="Time between prediction updates.",
        ),
        al.AgentVariable(
            name="comfort_interval",
            value=43200,
            description="Time between comfort updates.",
        ),
        al.AgentVariable(
            name="upper_comfort_high",
            value=295,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="upper_comfort_low",
            value=297,
            description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_high",
            value=292,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_low",
            value=290,
            description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="day_of_year",
            value=1,
            description="Day of the year for comfort schedule.",
        ),
        al.AgentVariable(
            name="year",
            value=2015,
            description="Year for comfort schedule.",
        ),
        al.AgentVariable(
            name="comfort_high_sleeping",
            value=295,
            description="High value in the comfort set point trajectory during sleeping.",
        ),
        al.AgentVariable(
            name="comfort_high_active",
            value=296,
            description="High value in the comfort set point trajectory during active periods.",
        ),
        al.AgentVariable(
            name="comfort_high_inactive",
            value=298,
            description="High value in the comfort set point trajectory during inactive periods.",
        ),
        al.AgentVariable(
            name="comfort_high_not_present",
            value=297,
            description="High value in the comfort set point trajectory when not present.",
        ),
        al.AgentVariable(
            name="slope",
            value=3,
            description="Slope of the comfort trajectory in K/h.",
        ),
        al.AgentVariable(
            name="comfort_low_sleeping",
            value=292,
            description="Low value in the comfort set point trajectory during sleeping.",
        ),
        al.AgentVariable(
            name="comfort_low_active",
            value=293,
            description="Low value in the comfort set point trajectory during active periods.",
        ),
        al.AgentVariable(
            name="comfort_low_inactive",
            value=295,
            description="Low value in the comfort set point trajectory during inactive periods.",
        ),
        al.AgentVariable(
            name="comfort_low_not_present",
            value=292,
            description="Low value in the comfort set point trajectory when not present.",
        ),
    ]
    #datentyp (siehe oben .Schedule) # Schedule for the day, with start and end times for each comfort mode
    schedule: Schedule = Field(default_factory=WoliszSchedule)

    flex_event: FlexEvent = Field( 
        FlexEvent(
            upper_boundary_shift=0.0,
            lower_boundary_shift=-0.0,
            gradient=0.0,
            start=900,
            end=7200
        )
    )

    

    shared_variable_fields: List[str] = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the ambient temp and comfort setpoint
    at a specified interval"""

    config: PredictorModuleConfig

    def __init__(self, *args, **kwargs): #
        super().__init__(*args, **kwargs)
      
        self.schedule = self.get_schedule()
        self.weekday = get_day_of_year(self.get("year").value, self.get("day_of_year").value)[1]
        self.flex_event = self.get_flex_event()
        self.grid_ub = self.set_upper_comfort_trajectory()[0] # Initialize grid for upper comfort trajectory
        self.grid_lb = self.set_lower_comfort_trajectory()[0] 
        self.duration_list_ub = self.set_lower_comfort_trajectory()[1]
        self.duration_list_lb = self.set_lower_comfort_trajectory()[1] # Initialize duration list for lower comfort trajectory
        self.traj_ub_adapted = self.set_upper_comfort_trajectory()[2] # Initialize upper comfort trajectory
        self.traj_lb_adapted = self.set_lower_comfort_trajectory()[2] # Initialize lower comfort trajectory
        
  
    def get_schedule(self):
        
        """Returns the schedule for the week"""
        schedule_type = self.config.schedule.type
        schedule_metadata = self.config.schedule.data
        # ?start_time_night = schedule_metadata["start_time"]?
        # switch case python

        # Define the schedule based on the type (e.g., Wolisz, default, etc.) -> look in the literature
        match schedule_type:
            case "wolisz" | "simple" | "test":
                return schedule_metadata
            case "default":
                return {}
            case _:
                # fallback for unknown types
                return schedule_metadata
            
    def get_flex_event(self):
        """Returns the flex event parameters"""
        flex_event = self.get("flex_event").value
        if isinstance(flex_event, FlexEvent):
            return flex_event.as_dict()
        else:
            return flex_event
   
    def register_callbacks(self):
        pass

    def process(self):
        """Sets a new prediction at each time step"""
        self.env.process(self.send_upper_comfort_trajectory())
        self.env.process(self.set_shifted_temperature())
        self.env.process(self.send_lower_comfort_trajectory())
        self.env.process(self.send_price_var_trajectory())
        
        
        while True:
            ts = self.get("prediction_sampling_time").value
            n = self.get("prediction_horizon").value
            now = self.env.now
            sample_time = self.get("sampling_time").value

            # temperature prediction   
            grid = np.arange(now, now + n * ts, ts)
            values = amb_temp_func(grid, uncertainty=0)
            traj = pd.Series(values, index=list(grid))
            self.set("T_amb", traj)
            yield self.env.timeout(sample_time)



    def set_upper_comfort_trajectory(self):

        slope = self.get("slope").value/3600  # Get the weekday name in K/s
        
        comfort_high_sleeping = self.get("comfort_high_sleeping").value
        comfort_high_active = self.get("comfort_high_active").value
        comfort_high_inactive = self.get("comfort_high_inactive").value
        comfort_high_not_present = self.get("comfort_high_not_present").value
        print(self.weekday)

       

        schedule = self.schedule[self.weekday]
        
        grid_ub = [start for start, _, _ in schedule]
        
        # Add the end time of the last schedule element if not already included
        
            
        values_ub = []
        duration_list_ub = []
            # Upper boundary comfort values

        for i, (start, end, comfort_mode) in enumerate(schedule):
            duration = end - start
            duration_list_ub.append(duration)
            
            # Determine start values for upper and lower boundaries
            if comfort_mode == "night":
                values_ub.append(comfort_high_sleeping)
                
                
            elif comfort_mode == "active":
                values_ub.append(comfort_high_active)
            
            elif comfort_mode == "not_present":
                values_ub.append(comfort_high_not_present)
                
            elif comfort_mode == "inactive":
                values_ub.append(comfort_high_inactive)

            if i == len(schedule) - 1:
                values_ub.append(values_ub[-1]) 
                grid_ub.append(end)

        
        
        traj_ub = pd.Series(values_ub, index=list(grid_ub))
        traj_ub_adapted = temp_trajectory_adapter(traj_ub)
        return grid_ub, duration_list_ub, traj_ub_adapted

    def send_upper_comfort_trajectory(self):

        while True:
            now = self.env.now

            for j in range(len(self.grid_ub)-1):
                traj_ub_send = self.traj_ub_adapted[self.traj_ub_adapted.index >= now]
                self.set("T_upper", traj_ub_send)
                
                # print(f"Sending upper comfort trajectory at time {now}: {traj_ub_send}")

                yield self.env.timeout(self.duration_list_ub[j]) # duration_list[j])  # Use the duration of the current schedule segment
                



    def set_lower_comfort_trajectory(self):


        slope = self.get("slope").value/3600  # Get the weekday name in K/s
       
        # Retrieve comfort values from PredictorModuleConfig
        comfort_low_sleeping = self.get("comfort_low_sleeping").value
        comfort_low_active = self.get("comfort_low_active").value
        comfort_low_inactive = self.get("comfort_low_inactive").value
        comfort_low_not_present = self.get("comfort_low_not_present").value

        
            

        """Sends the series for the comfort condition and logs it to a CSV file."""
            

        schedule = self.schedule[self.weekday]

        grid_lb = [start for start, _, _ in schedule]
        
        values_lb = []
        duration_list_lb = []
        for i, (start, end, comfort_mode) in enumerate(schedule):
            duration = end - start
            duration_list_lb.append(duration)
            
            # Determine start values for lower boundary
            if comfort_mode == "night":
                values_lb.append(comfort_low_sleeping)
                
            elif comfort_mode == "active":
                values_lb.append(comfort_low_active)
                
            elif comfort_mode == "not_present":
                values_lb.append(comfort_low_not_present)
                
            elif comfort_mode == "inactive":
                values_lb.append(comfort_low_inactive)
                
            if i == len(schedule) - 1:
                values_lb.append(values_lb[-1])  # Ensure the last value is repeated
                grid_lb.append(end)

        self.duration_list_lb = duration_list_lb
        
        traj_lb = pd.Series(values_lb, index=list(grid_lb))
        traj_lb_adapted = temp_trajectory_adapter(traj_lb)
        return grid_lb, duration_list_lb, traj_lb_adapted
    
    def send_lower_comfort_trajectory(self):

        while True:    
            now = self.env.now
            for j in range(len(self.grid_lb)-1):
                # traj_lb = pd.Series(values_lb, index=list(grid_lb))
                traj_lb_send = self.traj_lb_adapted[self.traj_lb_adapted.index >= now]
                self.set("T_lower", traj_lb_send)
                yield self.env.timeout(self.duration_list_lb[j])  # or duration_list_lb[j]

    def set_shifted_temperature(self):
        """
        sdfsdfsdf
        """
        
        while True: 
            now = self.env.now

            flex_event = self.flex_event
            flex_event_duration = flex_event["end"] - flex_event["start"]
            T_upper_series = self.traj_ub_adapted
            T_lower_series = self.traj_lb_adapted
            print(f"type of the T_upper: {type(T_upper_series)}") 
            print(f"T_upper_series: {T_upper_series}")     
            # create shifted temps
            temp_current_upper = temp_shift_traj_adapter(T_upper_series, now)
            temp_current_lower = temp_shift_traj_adapter(T_lower_series, now)

            shifted_upper_temp = temp_current_upper + flex_event["upper_boundary_shift"]
            shifted_lower_temp = temp_current_lower - flex_event["lower_boundary_shift"]
            
            self.set("T_upper_shadow", shifted_upper_temp)
            self.set("T_lower_shadow", shifted_lower_temp)
            print(f"Shifted upper temperature: {shifted_upper_temp}")
            print(f"Shifted lower temperature: {shifted_lower_temp}")
            yield self.env.timeout(flex_event_duration)  # Wait for the next comfort update
       

    def send_price_var_trajectory(self):
        """Sends the series for the price variable"""
        while True:
            ts = self.get("prediction_sampling_time").value
            n = self.get("prediction_horizon").value
            now = self.env.now
            sample_time = self.get("sampling_time").value

            grid = np.arange(now, now + n * ts, ts)
            traj = pd.Series([1 for i in grid], index=list(grid))
            self.set("r_pel", traj)
            yield self.env.timeout(sample_time)

def amb_temp_func(current, uncertainty):
    """Returns the ambient temperature in K, given a time in seconds"""
    value = np.zeros(shape=current.shape)
    for i in range(current.size):
        random_factor = 1 + uncertainty * (np.random.random() - 0.5)
        value[i] = random_factor * (278.15 + 5 * np.sin(2*np.pi * current[i] / 86400))
    return value


            # l = 0
            # for j in range(len(values_ub)-1):
            #     time_delta =  abs((values_ub[l+j+1] - values_ub[l+j])) / slope
            #     grid.insert(l + j + 2, grid[l + j+1] + time_delta)
            #     values_ub.insert(l +j + 1, values_ub[l + j])
            #     l+=1
            # values_ub.append(values_ub[-1])  # Ensure the last value is repeated

            # traj_ub = pd.Series(values_ub, index=list(grid))
            # for j in range(len(grid)-1):
            #     now = self.env.now
            #     traj_ub_send = traj_ub[traj_ub.index >= now]
            #     self.set("T_upper", traj_ub_send)
            #     # print(f"Sending upper comfort trajectory at time {now}: {traj_ub_send}")

            #     yield self.env.timeout(duration_list[j])  


        # def interpolation_compensator(self, values_ub, grid, slope):
        # """Compensates for the slope in the comfort trajectory by inserting intermediate values"""
        # """Inserts at the grid + time_delta the values in values_ub"""
        # traj_ub_inter = pd.Series(values_ub, index=list(grid))






        # l = 0
        # for j in range(len(values_lb)-1):
        #     time_delta = abs((values_lb[l+j+1] - values_lb[l+j])) / slope
        #     grid.insert(l + j + 2, grid[l + j+1] + time_delta)
        #     values_lb.insert(l + j + 1, values_lb[l + j])
        #     l += 1

        # values_lb.append(values_lb[-1])  # Ensure the last value is repeated

        # traj_lb = pd.Series(values_lb, index=list(grid))
        # for j in range(len(grid)-1):
        #     now = self.env.now
        #     traj_lb_send = traj_lb[traj_lb.index >= now]
        #     self.set("T_lower", traj_lb_send)

        #     print(f"Sending lower comfort trajectory at time {now}: {traj_lb_send}")
