import nhpp
import numpy as np 
import pandas as pd
from copy import deepcopy
from pickle import loads, dumps
import matplotlib.pyplot as plt

class Doctor:

    def __init__(self, shift_start, shift_end):
        self.shift_start = shift_start
        self.shift_end = shift_end
        self.patient = None
        self.end_time = float(shift_start)

    def __repr__(self):
        return f"Doctor({self.shift_start}-->{self.shift_end}): {self.end_time}"
    
    def __deepcopy__(self, memodict={}):
        doctor = Doctor(self.shift_start, self.shift_end)
        doctor.patient, doctor.end_time = self.patient, self.end_time
        return doctor
    
    def is_working(self, time):
        return True if time < self.shift_end and time >= self.shift_start else False
    
class Patient:

    def __init__(self, arrival_time, triage_level, service_rate, patience, by_ambulance=False, hosp_number=1):
        self.arrival_time = arrival_time
        self.triage = triage_level
        self.by_ambulance = by_ambulance
        self.hosp_number = hosp_number 
        self.dead = False
        self.wait_time = 0.
        self.assigned_doctor = False
        self.diverted = False
        self.in_queue = False

        self.B = np.random.uniform(2.5, 3.5)
        self.nu = np.random.uniform(1.5, 2.5)
        self.tol = np.random.random()      
        self.service_time = np.random.exponential(service_rate[self.triage - 1])
        self.death_time = inverse_mortality_func(u=np.random.random(), a=1., t=self.triage, B=self.B, nu=self.nu)
        self.patience_time = min(self.death_time, np.random.weibull(5)*patience[self.triage - 1])

    def __repr__(self):
        return f"Patient; {self.arrival_time:.2f} arrival; {self.wait_time:.2f} wait"
    
    def dies(self):
        d, a = float(self.diverted), 1. - float(self.assigned_doctor)
        if self.tol < mortality_func(x=self.wait_time, d=d, a=a, t=self.triage, B=self.B, nu=self.nu): return True
        return False

class SortedQueue:

    def __init__(self):
        self.doctors = []
        self.length = 0
        self.earliest_endtime = np.inf
    
    def __repr__(self):
        return self.doctors

    def __deepcopy__(self, memodict={}):
        doctors_queue = SortedQueue()
        doctors_queue.doctors, doctors_queue.length, doctors_queue.earliest_endtime = [deepcopy(d) for d in self.doctors], self.length, self.earliest_endtime
        return doctors_queue

    def append(self, doctor: Doctor):
        self.insort_left(doctor)
        self.earliest_endtime = min(self.earliest_endtime, doctor.end_time)
        self.length += 1
    
    def pop(self):
        doctor = self.doctors.pop(0)
        self.length -= 1
        if self.doctors == []:
            self.earliest_endtime = np.inf
        else:
            self.earliest_endtime = self.doctors[0].end_time
        return doctor
    
    def insort_left(self, doctor: Doctor):
        lo = 0
        hi = self.length
        while lo < hi:
            mid = (lo+hi)//2
            if self.doctors[mid].end_time < doctor.end_time: lo = mid+1
            else: hi = mid
        self.doctors.insert(lo, doctor)

class TriageQueue:

    def __init__(self, patients, dummy):
        
        self.level = [[] for _ in range(5)]
        self.level[0] = [p for p in patients if p.triage == 1] + [dummy]
        self.level[1] = [p for p in patients if p.triage == 2] + [dummy]
        self.level[2] = [p for p in patients if p.triage == 3] + [dummy]
        self.level[3] = [p for p in patients if p.triage == 4] + [dummy]
        self.level[4] = [p for p in patients if p.triage == 5] + [dummy]
        self.length = len(patients)
        self.current = [self.level[0][0].arrival_time,self.level[1][0].arrival_time,self.level[2][0].arrival_time,self.level[3][0].arrival_time,self.level[4][0].arrival_time]
        self.x = np.argmin(self.current)
        self.earliest_arrival_time = self.current[self.x]
        self.n_in_queue = 0
        self.n_arrived = 0 
        self.max_n_in_queue = 0

    def __deepcopy__(self, memodict={}):
        patients = [deepcopy(p) for level in self.level for p in level[:-1]]
        patients_queue = TriageQueue(patients, self.level[0][-1])
        patients_queue.length, patients_queue.current, patients_queue.x, patients_queue.earliest_arrival_time = self.length, self.current.copy(), self.x, self.earliest_arrival_time
        patients_queue.n_in_queue, patients_queue.n_arrived, patients_queue.max_n_in_queue = self.n_in_queue, self.n_arrived, self.max_n_in_queue
        return patients_queue

    def pop(self, l):
        try:
            self.current[l] = self.level[l][1].arrival_time
            self.x = np.argmin(self.current)
            self.earliest_arrival_time = self.current[self.x]
            self.length -= 1
            self.max_n_in_queue = max(self.max_n_in_queue, self.n_in_queue)
            self.n_in_queue -= 1
            return self.level[l].pop(0)
        except IndexError:
            return None
        
    def assign_patient(self, t, hn=1):
        # remove diverted patients
        for i in range(len(self.level)):
            p = self.level[i][0]
            while p.diverted and p.hosp_number == hn:
                self.pop(i)
                p = self.level[i][0]
        if self.length == 0: return None, False

        for i, c in enumerate(self.current):
            if c < t:
                self.abandoned_queue(i, t)
                return self.pop(i), False # return highest triage
        return self.pop(self.x), True # return earliest arrival
    
    def abandoned_queue(self, i, t):
        if i < 4:
            for j, c in enumerate(self.current[i + 1:]):
                patient = self.level[i + 1 + j][0]
                while patient.arrival_time + patient.patience_time < t:
                    patient.wait_time += patient.patience_time
                    patient.in_queue = False
                    self.pop(i + 1 + j)
                    patient = self.level[i + 1 + j][0]

    def insort_left(self, patients):
        for patient in patients:
            t = patient.triage - 1
            lo = 0
            hi = len(self.level[t]) - 1
            arrival_time = patient.arrival_time
            while lo < hi:
                mid = (lo + hi)//2
                if self.level[t][mid].arrival_time < patient.arrival_time: lo = mid + 1
                else: hi = mid
            self.level[t].insert(lo, patient)
            self.length += 1
            if arrival_time < self.current[t]: 
                self.current[t] = arrival_time
        self.x = np.argmin(self.current)
        self.earliest_arrival_time = self.current[self.x]

class Hospital:

    def __init__(
            self, 
            hourly_arrival_rates, 
            service_rate, 
            patience, 
            triage_probabilities, 
            triage_probabilities_crisis,
            weekly_crisis_factor,
            threshold, 
            days, 
            patients,
            doctor_shift_counts=[2, 2, 4, 2, 4, 1],
            hosp_number=1
            ):
        
        self.doctor_shift_counts = doctor_shift_counts
        self.hourly_arrival_rates = hourly_arrival_rates
        self.service_rate = service_rate
        self.patience = patience
        self.triage_probabilities = triage_probabilities
        self.triage_probabilities_crisis = triage_probabilities_crisis
        self.weekly_crisis_factor = weekly_crisis_factor
        self.patients = patients
        self.n = len(self.patients)
        self.threshold = threshold
        self.days = days
        self.hosp_number = hosp_number
        self.doctors, self.prev_shift_start_20 = generate_doctors(start=0, doctor_shift_counts=self.doctor_shift_counts)

        self.dummy = Patient(np.inf, 5, self.service_rate, self.patience)
        self.patients_queue = TriageQueue(self.patients, self.dummy)
        self.doctors_queue = SortedQueue()
    
    def simulate(self, summary=False):
        
        self.patients_queue = TriageQueue(self.patients, self.dummy)
        for d in range(self.days):
            for shift_start in [4*i + 24*d for i in range(6)]:
                next_shift_start = shift_start + 4
                for doctor in self.doctors[shift_start]:
                    self.doctors_queue.append(doctor)
                self.simulate_one_shift(next_shift_start, self.patients, self.doctors_queue, self.patients_queue, self.threshold, self.hosp_number)
            self.doctors, self.prev_shift_start_20 = generate_doctors(start=24*(d+1), prev_shift_start_20=self.prev_shift_start_20, doctor_shift_counts=self.doctor_shift_counts)
        
        if summary: self._summary()

    def sample_one_shift_ahead(self, patients_in_queue, shift_start, next_shift_start, threshold, diverted_patients=None, travel_time=1./60., hosp_number=1):
        # patients = deepcopy(patients_in_queue)
        patients = loads(dumps(patients_in_queue, -1))
        n_in_queue = len(patients)
        patients += generate_patients(self.hourly_arrival_rates[shift_start:next_shift_start + 1], self.service_rate, self.patience, 
                                      self.triage_probabilities, self.triage_probabilities_crisis, self.weekly_crisis_factor, 
                                      shift_start=shift_start, hosp_number=hosp_number) # one shift
        if diverted_patients: self.insort_diverted(diverted_patients, patients, travel_time)
        # doctors_queue = deepcopy(self.doctors_queue)
        doctors_queue = loads(dumps(self.doctors_queue, -1))
        patients_queue = TriageQueue(patients, self.dummy)
        patients_queue.n_in_queue = n_in_queue
        diverted_patients = self.simulate_one_shift(next_shift_start, patients, doctors_queue, patients_queue, threshold, hosp_number)
        return patients, diverted_patients
    
    def sample_till_end(
            self, 
            patients_in_queue, 
            doctors_, 
            shift_start, 
            remaining_shifts, 
            remaining_days, 
            threshold, 
            prev_shift_start_20_,
            doctor_shift_counts,
            diverted_patients=None, 
            travel_time=1./60., 
            hosp_number=1
            ):
        patients = loads(dumps(patients_in_queue, -1))
        n_in_queue = len(patients)
        patients += generate_patients(self.hourly_arrival_rates[shift_start:], self.service_rate, self.patience, 
                                      self.triage_probabilities, self.triage_probabilities_crisis, self.weekly_crisis_factor, 
                                      shift_start=shift_start, hosp_number=hosp_number) # one shift
        if diverted_patients: self.insort_diverted(diverted_patients, patients, travel_time)
        doctors_queue = loads(dumps(self.doctors_queue, -1))
        patients_queue = TriageQueue(patients, self.dummy)
        patients_queue.n_in_queue = n_in_queue
        doctors = loads(dumps(doctors_, -1))
        prev_shift_start_20 = loads(dumps(prev_shift_start_20_, -1))
    
        for shift_start in remaining_shifts:
            for doctor in doctors[shift_start]:
                doctors_queue.append(doctor)
            next_shift_start = shift_start + 4
            self.simulate_one_shift(next_shift_start, patients, doctors_queue, patients_queue, threshold, hosp_number)                
        prev_shift_start_20 = doctors[max(doctors.keys())][:doctor_shift_counts[-1]]

        for d in remaining_days:
            doctors, prev_shift_start_20 = generate_doctors(start=24*d, prev_shift_start_20=prev_shift_start_20, doctor_shift_counts=doctor_shift_counts)
            for shift_start in [4*i + 24*d for i in range(6)]:
                next_shift_start = shift_start + 4
                for doctor in doctors[shift_start]:
                    doctors_queue.append(doctor)
                self.simulate_one_shift(next_shift_start, patients, doctors_queue, patients_queue, threshold, hosp_number)
                
        return patients

    @staticmethod
    def simulate_one_shift(next_shift_start, patients, doctors_queue, patients_queue, threshold, hosp_number):

        diverted_patients = []

        while doctors_queue.length > 0:
            doctor = doctors_queue.pop()

            if patients_queue.length > 0 and doctor.end_time < next_shift_start:
                patient, earliest_arrival_assignment = patients_queue.assign_patient(doctor.end_time, hn=hosp_number)

                if patient:
                    lowest_triage = 1
                    patient.in_queue = False

                    if patient.triage == 1 and not earliest_arrival_assignment and doctor.patient:
                        lowest_triage = doctor.patient.triage
                        doctor_to_switch = doctor

                        for doctor_ in doctors_queue.doctors:

                            if doctor_.patient:
                                level = doctor_.patient.triage

                                if level > lowest_triage:
                                    doctor_to_switch, lowest_triage = doctor_, level

                        if lowest_triage != 1:
                            if patient.patience_time <= patient.wait_time:
                                patient.wait_time += patient.patience_time
                            else:
                                patient.assigned_doctor = True
                                doctor_to_switch.patient.service_time += patient.service_time
                                doctor_to_switch.end_time = doctor_to_switch.end_time + patient.service_time

                    if lowest_triage == 1:
                        doctor.patient = patient
                        wait_time = max(0., doctor.end_time - doctor.patient.arrival_time)

                        if doctor.patient.patience_time < wait_time:
                            doctor.patient.wait_time += doctor.patient.patience_time
                        else:
                            doctor.patient.wait_time += wait_time
                            doctor.end_time = max(doctor.end_time + patient.service_time, doctor.patient.arrival_time + doctor.patient.service_time)
                            doctor.patient.assigned_doctor = True

                    if doctor.is_working(doctor.end_time): doctors_queue.append(doctor)

                while patients[patients_queue.n_arrived].arrival_time < min(doctor.end_time, next_shift_start) and patients_queue.n_arrived < len(patients) - 1:
                    _patient = patients[patients_queue.n_arrived]
                    patients_queue.n_in_queue += 1
                    patients_queue.n_arrived += 1

                    if patients_queue.n_in_queue > threshold and _patient.by_ambulance and not _patient.assigned_doctor:
                        _patient.diverted = True
                        diverted_patients.append(_patient)
                    else:
                        _patient.in_queue = True and not _patient.assigned_doctor and not _patient.diverted
        
        return diverted_patients

    def insort_diverted(self, diverted_patients, patients_list=None, travel_time=1./60., in_triage_queue=False):
        if not patients_list: patients_list = self.patients
        n = len(patients_list)
        for patient in diverted_patients:
            patient.arrival_time += travel_time
            patient.wait_time += travel_time
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if patients_list[mid].arrival_time < patient.arrival_time: lo = mid + 1
                else: hi = mid
            patients_list.insert(lo, patient)
        if in_triage_queue:
            self.patients_queue.insort_left(diverted_patients)

    def _summary(self):
        print(len(self.patients[1000:-1000]), 
              len([p for p in self.patients[1000:-1000] if p.diverted]),
              len([p for p in self.patients[1000:-1000] if p.assigned_doctor == True]))
        print([len([p for p in self.patients[1000:-1000] if p.assigned_doctor == True and p.triage == t]) for t in range(1, 6)])
        print(np.mean([p.wait_time for p in self.patients[1000:-1000] if not p.diverted]))
        print(np.mean([p.wait_time for p in self.patients[1000:-1000] if p.assigned_doctor == True]))
        print([np.mean([p.wait_time for p in self.patients[1000:-1000] if p.triage == t and not p.diverted]) for t in range(1, 6)])
        print([np.mean([1 if p.assigned_doctor else 0 for p in self.patients[1000:-1000] if p.triage == t and not p.diverted]) for t in range(1, 6)])


def simulate_system_by_shift(hosp1, hosp2, days, reps=10, thresholds_to_test=[0, 25, 50, 75, 100, 125, 150, 175, 200], show_output=True, simulate_one_step_ahead=True):
    '''Hospital 2 receives Hospital 1's diverted patients'''

    res = np.zeros((days, 6))
    all_thresholds = np.repeat(hosp1.threshold, 6)
    nt = len(thresholds_to_test)
    for d in range(days):
        if d % 7 == 0 and show_output: print('Week:', d // 7)
        daily_shifts = [4*i + 24*d for i in range(6)]
        for j, shift_start in enumerate(daily_shifts):
            next_shift_start = shift_start + 4
            
            # simulate for best threshold 
            if simulate_one_step_ahead:
                mortality_rate = np.zeros(nt)
                patients_currently_in_queue1 = [p for p in hosp1.patients if p.in_queue and p.arrival_time < shift_start]
                patients_currently_in_queue2 = [p for p in hosp2.patients if p.in_queue and p.arrival_time < shift_start]
                print(len(patients_currently_in_queue1), len(patients_currently_in_queue2), end=' ')

                remaining_shifts = daily_shifts[j:]
                remaining_days = [di for di in range(d + 1, days)]
                doctors1, prev_shift_start_20_1 = loads(dumps(hosp1.doctors, -1)), loads(dumps(hosp1.prev_shift_start_20, -1))
                doctors2, prev_shift_start_20_2 = loads(dumps(hosp2.doctors, -1)), loads(dumps(hosp2.prev_shift_start_20, -1))

                for i, threshold in enumerate(thresholds_to_test):
                    for k in range(reps):

                        # hosp1_patients, diverted_patients = hosp1.sample_one_shift_ahead(patients_currently_in_queue1, shift_start, next_shift_start, threshold)
                        hosp1_patients = hosp1.sample_till_end(patients_currently_in_queue1, doctors1, shift_start, remaining_shifts, remaining_days,
                                                               threshold, prev_shift_start_20_1, hosp1.doctor_shift_counts)
                        diverted_patients = [p for p in hosp1_patients if p.diverted]
                        # hosp2_patients, _ = hosp2.sample_one_shift_ahead(patients_currently_in_queue2, shift_start, next_shift_start, np.inf, diverted_patients, hosp_number=2)
                        hosp2_patients = hosp2.sample_till_end(patients_currently_in_queue2, doctors2, shift_start, remaining_shifts, remaining_days,
                                                               np.inf, prev_shift_start_20_2, hosp2.doctor_shift_counts, diverted_patients)
                        # mortality_rate[i] += np.mean([p.dies() for p in hosp1_patients if not p.diverted] + [p.dies() for p in hosp2_patients]) / reps
                        mortality_rate[i] += np.mean([p.wait_time - p.death_time >= -1e-8 for p in hosp1_patients if not p.diverted] + 
                                                     [p.wait_time - p.death_time >= -1e-8 for p in hosp2_patients]) / reps
                best_threshold = thresholds_to_test[np.argmin(mortality_rate)]
                # all_thresholds[j] += (best_threshold - all_thresholds[j]) / (j + d*6 + 1)
                print(np.round(mortality_rate, 3))
                all_thresholds[j] = best_threshold

            for doctor in hosp1.doctors[shift_start]:
                hosp1.doctors_queue.append(doctor)

            for doctor in hosp2.doctors[shift_start]:
                hosp2.doctors_queue.append(doctor)

            diverted_patients = hosp1.simulate_one_shift(next_shift_start, hosp1.patients, hosp1.doctors_queue, hosp1.patients_queue, all_thresholds[j], hosp1.hosp_number)
            # diverted_patients = [p for p in hosp1.patients if p.diverted and p.arrival_time >= shift_start and p.arrival_time < next_shift_start]
            hosp2.insort_diverted(diverted_patients, in_triage_queue=True, travel_time=1./60.)
            hosp2.simulate_one_shift(next_shift_start, hosp2.patients, hosp2.doctors_queue, hosp2.patients_queue, hosp2.threshold, hosp2.hosp_number)

        hosp1.doctors, hosp1.prev_shift_start_20 = generate_doctors(start=24*(d+1), prev_shift_start_20=hosp1.prev_shift_start_20, doctor_shift_counts=hosp1.doctor_shift_counts)
        hosp2.doctors, hosp2.prev_shift_start_20 = generate_doctors(start=24*(d+1), prev_shift_start_20=hosp2.prev_shift_start_20, doctor_shift_counts=hosp2.doctor_shift_counts)
        if show_output: print(np.round(all_thresholds, 2))
        res[d] = all_thresholds
    if show_output: print(all_thresholds)
    return res

def generate_patients(
        hourly_arrival_rates, 
        service_rate, patience, 
        triage_probabilities, 
        triage_probabilities_crisis, 
        weekly_crisis_factor, 
        shift_start=0, 
        hosp_number=1
        ):
    knots = {i + shift_start: h for i, h in enumerate(hourly_arrival_rates)}
    arrival_times = nhpp.get_arrivals(knots)
    n = len(arrival_times)
    by_ambulance = np.random.choice([False, True], size=n, p=[0.75, 0.25])
    # triage = np.array([t for i, t in enumerate(np.random.choice([1, 2, 3, 4, 5], size=n, p=triage_probabilities))])
    triage = np.array([compute_triage(arrival_time, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor) 
                       for arrival_time in arrival_times])
    
    return [Patient(a, t, service_rate, patience, b, hosp_number=hosp_number) for a, t, b in zip(arrival_times, triage, by_ambulance)]

def generate_doctors(start, doctor_shift_counts=[2, 2, 4, 2, 4, 1], prev_shift_start_20=None):
    shift_start_0 = [Doctor(0 + start, 8 + start) for _ in range(doctor_shift_counts[0])]
    shift_start_4 = [Doctor(4 + start, 12 + start) for _ in range(doctor_shift_counts[1])]
    shift_start_8 = [Doctor(8 + start, 16 + start) for _ in range(doctor_shift_counts[2])]
    shift_start_12 = [Doctor(12 + start, 20 + start) for _ in range(doctor_shift_counts[3])]
    shift_start_16 = [Doctor(16 + start, 24 + start) for _ in range(doctor_shift_counts[4])]
    shift_start_20 = [Doctor(20 + start, 28 + start) for _ in range(doctor_shift_counts[5])]
    total_start_0 = shift_start_0 + prev_shift_start_20 if prev_shift_start_20 else shift_start_0
    return {0 + start: total_start_0, 
            4 + start: shift_start_4 + shift_start_0, 
            8 + start: shift_start_8 + shift_start_4, 
            12 + start: shift_start_12 + shift_start_8, 
            16 + start: shift_start_16 + shift_start_12,  
            20 + start: shift_start_20 + shift_start_16}, shift_start_20

def mortality_func(x, d, a, t, B, nu, K={1:1.,2:0.9,3:.05,4:.02,5:.01}, A={1:.6,2:.1,3:0.,4:0.,5:0.}):
    return A[t] + (K[t] - A[t]) / (1.+(3.*t) * np.exp(-(B+5-t)*(x) + (-2+2.5*a)*(t))) ** (1./(nu+0.25*t))

def inverse_mortality_func(u, a, t, B, nu, K={1:1.,2:0.9,3:.05,4:.02,5:.01}, A={1:.6,2:.1,3:0.,4:0.,5:0.}):
    if u <= mortality_func(0., d=0, a=a, t=t, B=B, nu=nu): return 0.
    elif u >= K[t]: return np.inf
    return -np.log((((K[t] - A[t]) / (u - A[t])) ** (nu+0.25*t) - 1.) / (3.*t)) / (B+5.-t) + (-2.+2.5*a)*t/(B+5.-t)

def plot_mortality(save=True):
    plt.rcParams['figure.figsize'] = (15, 4)
    x = np.linspace(0, 6, 1000)
    for t in [1, 2, 3, 4, 5]:
        y = mortality_func(x, d=0., a=1., t=t, B=3., nu=2.)
        y1 = mortality_func(x, d=0., a=1., t=t, B=3.5, nu=2.5)
        y2 = mortality_func(x, d=0., a=1., t=t, B=2.5, nu=1.5)
        plt.plot(x, y, label=f'triage={t}')
        plt.fill_between(x, y1, y2, alpha=0.1)
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Death time (hours)')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    if save: plt.savefig('mortality.png', bbox_inches='tight')
    plt.show()

def plot_arrival_rate(hourly_arrival_rates_, save=False):
    plt.rcParams["figure.figsize"] = (15, 5)
    data = hourly_arrival_rates_
    n = len(data)
    plt.plot(data)
    plt.xlabel('Day')
    plt.ylabel('Arrival Rate')
    if save: plt.savefig('arrival_rate.png', bbox_inches='tight')
    plt.xticks(np.arange(0,n+1,24), labels=np.arange(0, n//24 + 1))
    plt.show()

def plot_predicted(model, data, periods=24*7):
    n = len(data)
    fitted, confint = model.predict(n_periods=periods, return_conf_int=True)
    index_of_fc = np.arange(n, n + periods)

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(data, color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("Arrival rate forecast")
    plt.show()

def compute_triage(arrival_time, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor):
    week = int(arrival_time // (7*24))
    if weekly_crisis_factor[week] > 1.:
        return np.random.choice([1, 2, 3, 4, 5], size=1, p=triage_probabilities_crisis)[0]
    else:
        return np.random.choice([1, 2, 3, 4, 5], size=1, p=triage_probabilities)[0]
    
def single_threshold_simulation(
        patients_hosp1,
        patients_hosp2,
        hourly_arrival_rates, 
        service_rate, patience, 
        triage_probabilities, 
        triage_probabilities_crisis, 
        weekly_crisis_factor, 
        thresholds, 
        reps,
        days,
        hosp2_doctor_shift_counts,
        ):
    
    mortality_rates = [0. for _ in thresholds]
    for i, threshold in enumerate(thresholds):
        for k in range(reps):
            if reps == 1:
                patients_hosp1_c, patients_hosp2_c = loads(dumps(patients_hosp1, -1)), loads(dumps(patients_hosp2, -1))
            else:
                patients_hosp1_c = generate_patients(hourly_arrival_rates, service_rate, patience, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor, hosp_number=1)
                patients_hosp2_c = generate_patients(hourly_arrival_rates, service_rate, patience, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor, hosp_number=2)
            
            hosp1 = Hospital(hourly_arrival_rates, service_rate, patience, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor, threshold, days, patients_hosp1_c, hosp_number=1)
            hosp2 = Hospital(hourly_arrival_rates, service_rate, patience, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor, np.inf, days, patients_hosp2_c, doctor_shift_counts=hosp2_doctor_shift_counts, hosp_number=2)
            simulate_system_by_shift(hosp1, hosp2, days, show_output=False, simulate_one_step_ahead=False)
            mortality_rates[i] += np.mean([p.wait_time - p.death_time >= -1e-8 for p in hosp1.patients[100:-100] if not p.diverted] +
                                            [p.wait_time - p.death_time >= -1e-8 for p in hosp2.patients[100:-100]]) / reps
        print('Threshold:', threshold, '---', 'Mortality rate:', np.round(mortality_rates[i], 4))

    plt.plot(thresholds, mortality_rates)
    plt.xlabel('Thresholds')
    plt.ylabel('Mortality rate')
    plt.savefig('mortality_rates_thresholds.png', bbox_inches='tight')
    plt.show()

def varying_threshold_simulation(
        hourly_arrival_rates, 
        service_rate, 
        patience, 
        triage_probabilities, 
        triage_probabilities_crisis, 
        weekly_crisis_factor, days, 
        patients_hosp1, 
        patients_hosp2, 
        hosp2_doctor_shift_counts
        ):
    hosp1 = Hospital(hourly_arrival_rates, service_rate, patience, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor, np.inf, days, patients_hosp1, hosp_number=1)
    hosp2 = Hospital(hourly_arrival_rates, service_rate, patience, triage_probabilities, triage_probabilities_crisis, weekly_crisis_factor, np.inf, days, patients_hosp2, doctor_shift_counts=hosp2_doctor_shift_counts, hosp_number=2)
    optimal_thresholds = simulate_system_by_shift(hosp1, hosp2, days, show_output=True)
    print(np.mean([p.wait_time - p.death_time >= -1e-8 for p in hosp1.patients[100:-100] if not p.diverted] + 
                    [p.wait_time - p.death_time >= -1e-8 for p in hosp2.patients[100:-100]]))
    plt.rcParams['figure.figsize'] = (20, 5)
    plt.xlabel('Shift'); plt.ylabel('Threshold'); plt.plot(optimal_thresholds.ravel()); plt.savefig('varying_thresholds.png', bbox_inches='tight')