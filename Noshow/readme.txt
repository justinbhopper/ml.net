select
	a.AppointmentDate,
	a.AppointmentTime,
	l.Minutes,
	a.CreateDate,
	a.ShowTime,
	a.ShowNoShow,
	case when a.RecFirstKey is null then 1 else 0 end as IsRecurring,
	case when a.RecFirstKey = a.ID then 1 else 0 end as IsFirstInRecurrence,
	a.ClientKey,
	c.DateOfBirth,
	c.Sex,
	c.SexOrientKey,
	c.OMBWhite,
	c.OMBAmericanIndian,
	c.OMBAsian,
	c.OMBBlack,
	c.OMBHawaiian,
	e.CDCCode,
	case when isnull(c.EmergencyContName, '') <> '' then 1 else 0 end as HasEmergencyContact,
	lastAppt.ShowNoShow as LastAppointmentShowNoShow,
	totalNoShow.Total as PreviousNoShows,
	prevAppts.Total as TotalScheduled,
	lastApptMeds.Total as LastAppointmentScripts
from 
	dbo.Appointments a
	join dbo.SessionLength l on l.ID = a.LengthKey
	join dbo.Clients c on c.ID = a.ClientKey
	left join dbo.EnrollRace e
		on c.RaceCode = e.Ethnicity_Tribal_Code
		and NULLIF(LTRIM(RTRIM(e.CDCCode)),'') IS NOT NULL
		and
			(
				(e.EffectiveDate  IS NOT NULL and e.EffectiveDate  <= GETDATE())
			and (e.ExpirationDate IS     NULL or  e.ExpirationDate >  GETDATE())
			)
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
	and a.ShowNoShow is not null
	and a.AppointmentTime is not null
	and a.ClientKey is not null
	and c.DateOfBirth is not null
	and c.Sex is not null