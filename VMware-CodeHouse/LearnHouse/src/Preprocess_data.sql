select avg(age) from people;

select count(*) from people where age<40;

update train_people set is_age_more_40=1
from people
where people.age<=40
and people.id=train_people.id;


update train_people set education_is_G=0
from people
where trim(people.education) in ('Bachelors', 'Masters', 'Doctorate')
and people.id=train_people.id;

update train_people set employer_type_is_gov=0
from people
where trim(people.type_employer) in ('Federal-gov', 'Local-gov', 'State-gov')
and people.id=train_people.id;



update train_people set is_sex_male=0
from people
where trim(people.sex) = ('Male')
and people.id=train_people.id;


update train_people set is_avg_work_less_40=0
from people
where hours_per_week < 40
and people.id=train_people.id;