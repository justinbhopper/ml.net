https://scholar.smu.edu/cgi/viewcontent.cgi?article=1059&context=datasciencereview
https://www.linkedin.com/pulse/how-predict-no-show-analysis-appointment-rates-brazilian-burns/
https://journalofbigdata.springeropen.com/articles/10.1186/s40537-020-00384-9
https://www.ideas2it.com/blogs/patient-appointment-no-shows-prediction/
https://medium.com/docplanner-tech/predicting-cancellations-in-visit-by-appointment-systems-with-tensorflow-d6b78f8621ff
https://github.com/adelweiss/MedicalAppt

select
	case a.ShowNoShow 
		when 2 then 1 -- No-show considered NoShow
		when 3 then 1 -- Cancelled also considered NoShow
		else 0 
	end as NoShow,
	datediff(day, a.CreateDate, a.AppointmentDate) as LeadTime,
	datepart(weekday, a.AppointmentDate) as DayOfWeek,
	datepart(hour, a.AppointmentTime) as Hour,
	datediff(year, c.DateOfBirth, a.AppointmentDate) as Age,
	totalNoShow.Total as PreviousNoShows,
	prevAppts.Total as TotalScheduled
from 
	dbo.Appointments a
	join dbo.SessionLength l on l.ID = a.LengthKey
	join dbo.Clients c on c.ID = a.ClientKey
	outer apply (
		select COUNT(*) as Total
		from dbo.Appointments a2 
		where 
			a2.ID != a.ID
			and a2.ClientKey = a.ClientKey
			and a2.AppointmentDate < a.AppointmentDate
			and a2.ShowNoShow in (2, 3)
	) as totalNoShow
	outer apply (
		select COUNT(*) as Total
		from dbo.Appointments a3 
		where 
			a3.ID != a.ID
			and a3.ClientKey = a.ClientKey
			and a3.AppointmentDate < a.AppointmentDate
	) as prevAppts
where
	a.AppointmentDate is not null
	and a.CreateDate is not null
	and a.ShowNoShow in (1,2,3)
	and a.AppointmentTime is not null
	and a.ClientKey is not null
	and c.DateOfBirth is not null
	and l.Minutes is not null