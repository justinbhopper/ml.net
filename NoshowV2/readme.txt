https://scholar.smu.edu/cgi/viewcontent.cgi?article=1059&context=datasciencereview
https://www.linkedin.com/pulse/how-predict-no-show-analysis-appointment-rates-brazilian-burns/
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00384-9
https://www.ideas2it.com/blogs/patient-appointment-no-shows-prediction/

select
	case a.ShowNoShow when 2 then 1 else 0 end as NoShow,
	datediff(day, a.CreateDate, a.AppointmentDate) as LeadTime,
	datepart(weekday, a.AppointmentDate) as DayOfWeek,
	datepart(month, a.AppointmentDate) as Month,
	datepart(WEEK, a.AppointmentDate) as Week,
	datepart(hour, a.AppointmentDate) as Hour,
	l.Minutes,
	case when a.RecFirstKey is null then 1 else 0 end as IsRecurring,
	case when a.RecFirstKey = a.ID then 1 else 0 end as IsFirstInRecurrence,
	datediff(year, c.DateOfBirth, a.AppointmentDate) as Age,
	case c.Sex when 'M' then 1 else 0 end as Male,
	COALESCE(c.OMBWhite, 0) as OMBWhite,
	COALESCE(c.OMBAmericanIndian, 0) as OMBAmericanIndian,
	COALESCE(c.OMBAsian, 0) as OMBAsian,
	COALESCE(c.OMBBlack, 0) as OMBBlack,
	COALESCE(c.OMBHawaiian, 0) as OMBHawaiian,
	case when isnull(c.EmergencyContName, '') <> '' then 1 else 0 end as HasEmergencyContact,
	case lastAppt.ShowNoShow when 2 then 1 else 0 end as LastAppointmentNoShow,
	totalNoShow.Total as PreviousNoShows,
	prevAppts.Total as TotalScheduled,
	case prevAppts.Total when 0 then 0 else (totalNoShow.Total / CONVERT(decimal(7,2), prevAppts.Total)) end as NoShowRatio,
	lastApptMeds.Total as LastAppointmentScripts
from 
	dbo.Appointments a
	join dbo.SessionLength l on l.ID = a.LengthKey
	join dbo.Clients c on c.ID = a.ClientKey
	outer apply (
		select top 1 na.AppointmentDate, na.ShowNoShow
		from dbo.Appointments na 
		where 
			na.ID != a.ID
			and na.ClientKey = a.ClientKey
			and na.AppointmentDate < a.AppointmentDate
		order by datediff(day, na.AppointmentDate, a.AppointmentDate) asc, datediff(minute, na.AppointmentTime, a.AppointmentTime) asc
	) as lastAppt
	outer apply (
		select COUNT(*) as Total
		from dbo.Appointments na 
		where 
			na.ID != a.ID
			and na.ClientKey = a.ClientKey
			and na.AppointmentDate < a.AppointmentDate
			and ShowNoShow = 2
	) as totalNoShow
	outer apply (
		select COUNT(*) as Total
		from dbo.Appointments na 
		where 
			na.ID != a.ID
			and na.ClientKey = a.ClientKey
			and na.AppointmentDate < a.AppointmentDate
	) as prevAppts
	outer apply (
		select COUNT(*) as Total
		from dbo.MMScripts s 
		where 
			s.ClientKey = a.ClientKey
			and s.ScriptDate is not null
			and datediff(day, s.ScriptDate, lastAppt.AppointmentDate) between 0 and 3
	) as lastApptMeds
where
	a.AppointmentDate is not null
	and a.CreateDate is not null
	and a.ShowNoShow is not null
	and a.AppointmentTime is not null
	and a.ClientKey is not null
	and c.DateOfBirth is not null
	and l.Minutes is not null
	and (c.Sex = 'M' or c.Sex = 'F')
	and (a.ShowNoShow = 1 or a.ShowNoShow = 2)