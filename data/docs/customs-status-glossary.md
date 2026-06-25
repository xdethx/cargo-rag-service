# Customs & Shipment Status Glossary

This document explains the shipment status codes used by the MakroShip tracking system.
Each entry lists the code, what it means, and the typical time a shipment stays in that
state. All durations are business days unless stated otherwise. (Sample data — for testing.)

## Pre-transit statuses

- **REG — Registered.** The shipment has been created in the system and a tracking number
  has been issued, but the parcel has not yet been handed to a carrier. Typical duration:
  0–1 day.
- **PCK — Picked Up.** The carrier has collected the parcel from the sender. Typical
  duration: same day.
- **DEP — Departed Origin.** The parcel has left the origin facility and is moving toward
  the export hub. Typical duration: 1–2 days.

## In-transit statuses

- **EXH — At Export Hub.** The parcel is at the origin-country export hub awaiting an
  international leg (flight, vessel, or road line-haul). Typical duration: 1–3 days.
- **ITR — In Transit.** The parcel is on an international leg between the origin and
  destination countries. Typical duration: 2–9 days depending on lane and mode.
- **ARR — Arrived Destination Country.** The parcel has landed in the destination country
  and is waiting to be presented to customs. Typical duration: 1–2 days.

## Customs statuses

- **CUS-SUB — Submitted to Customs.** The declaration has been lodged with the destination
  customs authority and is awaiting assessment. Typical duration: 1–2 days.
- **CUS-HOLD — Held at Customs.** Customs has paused the shipment for a documentary review,
  usually because information is missing or inconsistent. The most common cause is a missing
  or vague commercial invoice. Typical duration: 2–5 days.
- **CUS-INSP — Customs Inspection.** Customs has selected the parcel for a physical
  inspection. The box is opened and the contents are checked against the declaration.
  Typical duration: 3–7 days.
- **DUTY-DUE — Duty / Tax Payable.** Customs has assessed import duty and/or VAT. The
  shipment will not be released until the charges are paid by the importer. Typical
  duration: held until paid, then 1 day to release.
- **CLR — Cleared.** Customs has released the shipment. It now moves to final-mile delivery.
  Typical duration: same day.

## Delivery statuses

- **OFD — Out for Delivery.** The parcel is on the delivery vehicle for the final leg.
  Typical duration: same day.
- **DLV — Delivered.** The parcel has been handed to the recipient and a delivery scan (or
  signature) has been recorded. This is a terminal status.

## Exception statuses

- **EXC — Exception.** A generic problem has interrupted the shipment (bad address, weather,
  damaged label, failed delivery attempt). The tracking note usually contains the specific
  reason. Typical duration: 1–3 days until resolved or escalated.
- **DMG — Damaged.** The parcel was damaged in transit and pulled from the network for
  assessment. A claim may be required. Typical duration: 2–10 days.
- **RTN — Returned to Sender.** Delivery failed permanently (refused, unpaid duty, or
  undeliverable address) and the parcel is being sent back to the origin. Typical duration:
  follows the original transit time in reverse.

## Quick reference

A normal international shipment usually moves REG → PCK → DEP → EXH → ITR → ARR → CUS-SUB →
CLR → OFD → DLV. The customs stage (CUS-*) is where most delays happen, and DUTY-DUE is the
status that depends on the importer rather than the carrier.
