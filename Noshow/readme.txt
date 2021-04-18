select top 100
	a.AppointmentDate,
	a.AppointmentTime,
	l.Minutes,
	a.CreateDate,
	a.ShowTime,
	a.ShowNoShow,
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
	case when isnull(c.EmergencyContName, '') <> '' then 1 else 0 end as HasEmergencyContact
from dbo.Appointments a
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